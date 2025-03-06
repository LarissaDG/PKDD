import os
import time
import torch
import numpy as np
import PIL.Image
import torchvision
import pandas as pd
import sys

from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor

# Função auxiliar para construir o prompt com o template SFT
def get_prompt(vl_chat_processor, prompt_text, is_token_based=True, use_alternate=False):
    conversation = [
        {"role": "User", "content": prompt_text},
        {"role": "Assistant", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
         conversations=conversation,
         sft_format=vl_chat_processor.sft_format,
         system_prompt="",
    )
    return sft_format + (vl_chat_processor.image_start_tag if is_token_based else vl_chat_processor.image_gen_tag)

# Função de geração para os modelos token-based
def generate_token_based(mmgpt, vl_chat_processor, prompt, temperature=1.0, parallel_size=1, cfg_weight=5, 
                         image_token_num=576, img_size=384, patch_size=16, save_path="generated.jpg"):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)
    
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
            
    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num), dtype=torch.int).cuda()
    past_key_values = None
    for i in range(image_token_num):
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        hidden_states = outputs.last_hidden_state
        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)
        next_token = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(1)
    
    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(torch.int), shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
    image = PIL.Image.fromarray(dec[0])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

# Função principal que processa o CSV e gera as imagens
# Preciso de um df que tenha as descrições ("Original, Positiva, Muito Positiva, Negativa, Muito negativa")
# Gerar imagens e novos df frames "Positiva/Negativa" "Muito Positiva/Muito negativa" 
def main():
    if len(sys.argv) != 2:
        print("Uso: python code.py <caminho_do_csv>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)
    
    print("Executando modelo SMALL (Janus-Pro-1B)...")
    small_model_path = "deepseek-ai/Janus-Pro-1B"
    small_processor = VLChatProcessor.from_pretrained(small_model_path)
    small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True)
    small_model = small_model.to(torch.bfloat16).cuda().eval()
    
    output_dir_small = os.path.join(os.path.dirname(input_csv), "generated_images_small")
    os.makedirs(output_dir_small, exist_ok=True)
    
    generated_filenames_small = []
    start_time_small = time.time()
    for idx, row in df.iterrows():
        prompt_text = row["Description"]
        prompt_small = get_prompt(small_processor, prompt_text, is_token_based=True, use_alternate=False)
        out_filename = os.path.join(output_dir_small, f"img_small_{idx}.jpg")
        generate_token_based(small_model, small_processor, prompt_small, parallel_size=1, save_path=out_filename)
        generated_filenames_small.append(out_filename)
        print(f"[SMALL] Processado prompt {idx}")
    end_time_small = time.time()
    print(f"Tempo total de geração (SMALL): {end_time_small - start_time_small:.2f} segundos")
    
    df['generated_filename'] = generated_filenames_small
    df.to_csv(input_csv, index=False)
    
if __name__ == "__main__":
    main()
