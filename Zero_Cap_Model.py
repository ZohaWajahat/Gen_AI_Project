import torch.nn.functional as F
from transformers import GPT2LMHeadModel
import open_clip
import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from Data_Processing.preprocess import create_coco_reference_map
import tqdm
import json
from Data_Processing.preprocess import preprocess_image

# --- CONFIGURATION ---
DATA_DIR = "./data/coco"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VAL_ANNOTATIONS_PATH = f'{DATA_DIR}/annotations/annotations/captions_val2017.json'

COCO_VAL_GT_JSON, COCO_VAL_GT_DICT = create_coco_reference_map(VAL_ANNOTATIONS_PATH)

# 1. Original ZeroCap Baseline (GPT-2 Medium + CLIP ViT-B/32)
LM_ORIG_NAME = "gpt2-medium"
VSM_ORIG_NAME = 'ViT-B-32'
tokenizer_orig = GPT2Tokenizer.from_pretrained(LM_ORIG_NAME)
# **THE FIX:** Assign EOS token as the PAD token for GPT-2
if tokenizer_orig.pad_token is None:
    tokenizer_orig.pad_token = tokenizer_orig.eos_token 
_, _, preprocess_orig = open_clip.create_model_and_transforms(VSM_ORIG_NAME, pretrained='openai')

# 2. New Hybrid Pipeline (SOTA LM + Adapted VSM e.g., Mistral-7B + ViT-L/14)
LM_HYBRID_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
VSM_HYBRID_NAME = 'ViT-L-14'
tokenizer_hybrid = AutoTokenizer.from_pretrained(LM_HYBRID_NAME)
if tokenizer_hybrid.pad_token is None:
    tokenizer_hybrid.pad_token = tokenizer_hybrid.eos_token 
_, _, preprocess_hybrid = open_clip.create_model_and_transforms(VSM_HYBRID_NAME, pretrained='openai')

print("Model tokenizers and VSM preprocessors initialized.")

# Initialize the VSM models again (as they are needed for inference)
vsm_model_orig, _, _ = open_clip.create_model_and_transforms(VSM_ORIG_NAME, pretrained='openai')
vsm_model_orig.to(DEVICE).eval()
vsm_model_hybrid, _, _ = open_clip.create_model_and_transforms(VSM_HYBRID_NAME, pretrained='openai')
vsm_model_hybrid.to(DEVICE).eval()

def get_vsm_embed(image_path, vsm_model, preprocess_func):
    """Encodes the image using the VSM (CLIP)."""
    image_tensor = preprocess_image(image_path, pipeline='original') # Uses the correct preprocessing
    with torch.no_grad():
        # NOTE: We use VSM_ORIG_NAME model (ViT-B/32) for the baseline
        return vsm_model.encode_image(image_tensor).squeeze(0)

def get_clip_score_single(text, image_embed, vsm_model):
    """Calculates the CLIP score (cosine similarity) between text and image embed."""
    with torch.no_grad():
        text_tokens = open_clip.tokenize([text]).to(DEVICE)
        text_embed = vsm_model.encode_text(text_tokens).squeeze(0)
        
        # Normalize and compute cosine similarity
        text_embed = F.normalize(text_embed.float(), p=2, dim=0)
        image_embed = F.normalize(image_embed.float(), p=2, dim=0)

        # Scale and move to CPU for list collection
        return (torch.dot(text_embed, image_embed) * 100).cpu().item()

# --- ZeroCap Approximation Function ---

# 1. Initialize Original LM
model_orig = GPT2LMHeadModel.from_pretrained(LM_ORIG_NAME).to(DEVICE).eval()

def generate_original_zerocap_approx(image_path, image_id, prompt="A photo of a", max_tokens=30, lambda_clip=1.0, top_k=50):
    """
    ZeroCap approximation: Uses CLIP score to modify GPT-2's next-token logits 
    at each step, simulating the guidance and context cache optimization.
    """
    image_embed = get_vsm_embed(image_path, vsm_model_orig, preprocess_orig)
    input_ids = tokenizer_orig.encode(prompt, return_tensors='pt').to(DEVICE)
    current_tokens = input_ids.tolist()[0]
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 1. Get standard LM logits (L_CE prior)
            outputs = model_orig(torch.tensor([current_tokens]).to(DEVICE))
            logits = outputs.logits[0, -1, :]
            
            # 2. Get top K candidate tokens
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            
            # 3. Calculate CLIP score for each candidate text
            clip_scores = []
            candidate_texts = []
            for token_id in top_k_indices.tolist():
                candidate_ids = current_tokens + [token_id]
                candidate_texts.append(tokenizer_orig.decode(candidate_ids, skip_special_tokens=True))
            
            # Calculate CLIP scores efficiently
            clip_scores = [get_clip_score_single(text, image_embed, vsm_model_orig) for text in candidate_texts]
            clip_scores = torch.tensor(clip_scores).to(DEVICE)

            # 4. Scale CLIP scores back into "logits" (L_CLIP)
            # Create a logit vector where only top-k positions are updated
            clip_logits = torch.full_like(logits, -1e9).to(DEVICE)
            clip_logits.scatter_(0, top_k_indices, clip_scores * lambda_clip)
            
            # 5. Guided Logits = Log_Probs(LM) + CLIP Guidance
            # Note: We skip the L_CE regularization term (KL divergence) approximation for simplicity here,
            # focusing on the core logit modification technique used in many ZeroCap follow-ups.
            guided_logits = F.log_softmax(logits, dim=-1) + clip_logits
            
            probs = F.softmax(guided_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            current_tokens.append(next_token)
            if next_token == tokenizer_orig.eos_token_id:
                break
                
    final_caption = tokenizer_orig.decode(current_tokens, skip_special_tokens=True).strip()
    return {"image_id": image_id, "caption": final_caption}

# # --- Full Baseline Run (Conceptual) ---
# Loop through all COCO images and save results to JSON for evaluation
def run_baseline_inference(output_file, image_list):
    results = []
    for img_info in tqdm.tqdm(image_list, desc="Running Original ZeroCap Baseline"):
        img_path = f"{DATA_DIR}/val2017/{img_info['file_name']}"
        caption_data = generate_original_zerocap_approx(img_path, img_info['id'])
        results.append(caption_data)
    
    with open(output_file, 'w') as f:
        json.dump(results, f)
run_baseline_inference('./results/original_zerocap_final.json', COCO_VAL_GT_JSON['images'])