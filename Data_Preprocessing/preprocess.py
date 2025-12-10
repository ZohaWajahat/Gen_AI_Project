import torch
from transformers import AutoTokenizer, GPT2Tokenizer
import open_clip 
from PIL import Image
import json
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "../data/coco"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# --- 2. Image Preprocessing Function ---
def preprocess_image(image_path, pipeline='hybrid'):
    """Loads and transforms an image for the respective VSM."""
    img = Image.open(image_path).convert("RGB")
    if pipeline == 'original':
        processed_tensor = preprocess_orig(img)
    elif pipeline == 'hybrid':
        processed_tensor = preprocess_hybrid(img)
    else:
        raise ValueError("Invalid pipeline specified. Choose 'original' or 'hybrid'.")
    return processed_tensor.unsqueeze(0).to(DEVICE)

# --- 3. Text Preprocessing (Tokenization) ---
def preprocess_text_for_lm(text, pipeline='hybrid', max_length=50):
    """Tokenizes text for the specified Language Model."""
    if pipeline == 'original':
        tokenizer = tokenizer_orig
    elif pipeline == 'hybrid':
        tokenizer = tokenizer_hybrid
    else:
        raise ValueError("Invalid pipeline specified. Choose 'original' or 'hybrid'.")
        
    encoding = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    return encoding['input_ids'].to(DEVICE)

# --- 4. Reference Preprocessing (Utility Functions) ---
def create_coco_reference_map(annotations_path):
    """Loads COCO annotations and formats them for the pycocoevalcap library."""
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    coco_eval_json = []
    gts_dict = {} 
    
    for anno in data['annotations']:
        img_id = str(anno['image_id'])
        caption = anno['caption']
        if img_id not in gts_dict: gts_dict[img_id] = []
        gts_dict[img_id].append(caption)
        
        unique_id = int(img_id) * 10 + len(gts_dict[img_id]) - 1 
        coco_eval_json.append({"image_id": int(img_id), "caption": caption, "id": unique_id})

    final_json = {"annotations": coco_eval_json, "images": data.get('images', [])}
    return final_json, gts_dict

print("Preprocessing functions defined.")

# --- Execution and Verification ---
VAL_ANNOTATIONS_PATH = f'{DATA_DIR}/annotations/annotations/captions_val2017.json'

COCO_VAL_GT_JSON, COCO_VAL_GT_DICT = create_coco_reference_map(VAL_ANNOTATIONS_PATH)

COCO_REF_MAP_PATH = './coco_val_refs.json'
with open(COCO_REF_MAP_PATH, 'w') as f:
    json.dump(COCO_VAL_GT_JSON, f)

print("\n--- Preprocessing Verification ---")
print(f"COCO References saved to: {COCO_REF_MAP_PATH}")

try:
    example_img_path = f"{DATA_DIR}/val2017/{COCO_VAL_GT_JSON['images'][0]['file_name']}"
    
    tensor_orig = preprocess_image(example_img_path, pipeline='original')
    tensor_hybrid = preprocess_image(example_img_path, pipeline='hybrid')
    
    print(f"\nOriginal (ViT-B/32) Tensor Shape: {tensor_orig.shape}")
    print(f"Hybrid (ViT-L/14) Tensor Shape: {tensor_hybrid.shape}")
    
    # This section should now run without error:
    text = "A cat sitting on a bright red motorcycle."
    tokens_orig = preprocess_text_for_lm(text, pipeline='original')
    tokens_hybrid = preprocess_text_for_lm(text, pipeline='hybrid')
    print(f"Original (GPT-2) Token Count: {tokens_orig.shape[1]}")
    print(f"Hybrid (Mistral) Token Count: {tokens_hybrid.shape[1]}")

except FileNotFoundError:
    print("\nWarning: COCO image files not found. Ensure the download/extraction was successful.")