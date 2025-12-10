from transformers import AutoModelForCausalLM, AutoModel
from torch import nn
import open_clip
import torch
from Data_Processing.preprocess import preprocess_image
from transformers import AutoTokenizer, GPT2Tokenizer
from Data_Processing.preprocess import create_coco_reference_map
import json
import tqdm

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

# --- Stage 1 & 2: VSM Adaptation and Hybrid Inference Setup ---

# 1. Load Hybrid Model (Requires a model that can accept a prefix/soft-prompt)
class VisionLanguageModel(nn.Module):
    def __init__(self, lm_name, vsm_name):
        super().__init__()
        # Load pre-trained components
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        # We load VSM as AutoModel to use its encoder for features
        self.vsm = open_clip.create_model(vsm_name, pretrained='openai') 
        # Simple projection layer to map VSM embedding dimension to LM embedding dimension
        # (This is the D_Adapt training component)
        self.projector = nn.Linear(self.vsm.visual.output_dim, self.lm.config.hidden_size)
    
    def forward(self, input_ids, pixel_values):
        # 1. Encode Image
        image_embed = self.vsm.encode_image(pixel_values)
        # 2. Project feature to LM space (soft prompt)
        prefix_embeds = self.projector(image_embed).unsqueeze(1) # [B, 1, Hidden_Size]

        # 3. Concatenate prefix with token embeddings
        token_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        
        # Standard LM forward pass
        return self.lm(inputs_embeds=inputs_embeds, return_dict=True)

# Here we initialize the model structure.
hybrid_model = VisionLanguageModel(LM_HYBRID_NAME, VSM_HYBRID_NAME).to(DEVICE).eval()
hybrid_model.load_state_dict(torch.load('./models/Hybrid_RL_Best.pt')) # Assume checkpoint loaded

def generate_hybrid_pipeline(image_path, image_id, hybrid_model, prompt_text=""):
    """
    Generates captions using the fine-tuned Hybrid VLM.
    """
    image_tensor = preprocess_image(image_path, pipeline='hybrid')
    
    # 1. Encode image and get projected soft prompt
    with torch.no_grad():
        image_embed = hybrid_model.vsm.encode_image(image_tensor)
        prefix_embeds = hybrid_model.projector(image_embed).unsqueeze(1)

        # 2. Encode text prompt
        input_ids = tokenizer_hybrid.encode(prompt_text, return_tensors='pt').to(DEVICE)
        
        # 3. Concatenate prefix and token embeddings
        token_embeds = hybrid_model.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
    
        # 4. Generate text using the fine-tuned model
        output = hybrid_model.lm.generate(
            inputs_embeds=inputs_embeds,
            max_length=50, # Standard length for COCO
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer_hybrid.eos_token_id
        )
    
    # The output includes the soft prompt (which is just noise when decoded) and the input prompt, so we decode and strip the unnecessary parts.
    generated_text = tokenizer_hybrid.decode(output[0, inputs_embeds.shape[1]:], skip_special_tokens=True).strip()
    
    return {"image_id": image_id, "caption": generated_text}

# # --- Full Hybrid Run (Conceptual) ---
# # Loop through all COCO images and save results to JSON for evaluation
def run_hybrid_inference(output_file, image_list):
    results = []
    for img_info in tqdm.tqdm(image_list, desc="Running Hybrid Pipeline"):
        img_path = f"{DATA_DIR}/val2017/{img_info['file_name']}"
        caption_data = generate_hybrid_pipeline(img_path, img_info['id'], hybrid_model)
        results.append(caption_data)
    
    with open(output_file, 'w') as f:
        json.dump(results, f)
run_hybrid_inference('./results/hybrid_final.json', COCO_VAL_GT_JSON['images'])