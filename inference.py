import torch
from transformers import AutoConfig
from train import DiffusionDenoisingModel, generate_text_from_amr
from amr_dataset import DocAMRDataset, docamr_collate_fn
from torch.utils.data import DataLoader
import os

base_model_name = "roberta-base"
denoiser_config = AutoConfig.from_pretrained(base_model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"

input_file = "./generation/docamr_docAMR.out"

print(f"Loading validation dataset from {input_file}...")
val_dataset = DocAMRDataset(input_file, base_model_name=base_model_name, max_seq_len=128)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=docamr_collate_fn)

# Instantiate the model with the dataset's isolated num_relations
model = DiffusionDenoisingModel(denoiser_config, node_dim=768, hidden_dim=768, num_relations=val_dataset.num_relations)

best_checkpoint_path = "./checkpoints/docamr_diffusion_best.pt"
print(f"Loading best checkpoint from {best_checkpoint_path}...")
checkpoint = torch.load(best_checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

print("Starting Generation Process...")
all_generated_texts = []

for batch_idx, batch in enumerate(val_dataloader):
    print(f"Processing batch {batch_idx + 1}/{len(val_dataloader)}...")
    
    # We pass the dataloader dictionary natively to the reverse loop sampling method
    generated_texts = generate_text_from_amr(
        model=model, 
        base_model_name=base_model_name, 
        amr_batch=batch, 
        device=device
    )
    
    for i, text in enumerate(generated_texts):
        print(f"\n--- Generated Document {batch_idx * 4 + i + 1} ---")
        print(text)
        all_generated_texts.append(text)

# Save inference outputs nicely
os.makedirs("./generation", exist_ok=True)
output_file = "./generation/generated_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for text in all_generated_texts:
        f.write(text + "\n\n")

print(f"\nFinished! Saved all generated texts to {output_file}")