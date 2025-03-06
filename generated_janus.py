import os
import time
import torch
import numpy as np
import PIL.Image
import torchvision
import pandas as pd
import sys

from transformers import AutoModelForCausalLM, AutoModel
from janus.models import MultiModalityCausalLM, VLChatProcessor

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Função auxiliar para construir o prompt com o template SFT
def get_prompt(vl_chat_processor, prompt_text, is_token_based=True, use_alternate=False):
    if use_alternate:
        conversation = [
            {"role": "<|User|>", "content": prompt_text},
            {"role": "<|Assistant|>", "content": ""},
        ]
    else:
        conversation = [
            {"role": "User", "content": prompt_text},
            {"role": "Assistant", "content": ""},
        ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
         conversations=conversation,
         sft_format=vl_chat_processor.sft_format,
         system_prompt="",
    )
    # Para os modelos token-based usamos image_start_tag; para difusão, image_gen_tag
    return sft_format + (vl_chat_processor.image_start_tag if is_token_based else vl_chat_processor.image_gen_tag)

# Função de geração para os modelos token-based ("small" e "big")
@torch.inference_mode()
def generate_token_based(mmgpt, vl_chat_processor, prompt, temperature=1.0, parallel_size=1, cfg_weight=5, image_token_num=576, img_size=384, patch_size=16, save_path="generated_samples/img_token.jpg"):
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

# Função principal que processa o CSV e gera as imagens para cada modelo
def main():
    print("Começo do script")
    # Configurações do e-mail
    seu_email = "codigosecreto2025@gmail.com"
    senha = "bpij xclh mwqr sofe"
    destinatario = "laladg18@gmail.com"

    # Criando a mensagem
    msg = MIMEMultipart()
    msg["From"] = seu_email
    msg["To"] = destinatario
    msg["Subject"] = "Fim execução generated_janus.py"

    corpo = "Fim do script Python"
    msg.attach(MIMEText(corpo, "plain"))

    # Caminho do CSV de entrada (cada linha possui uma descrição/prompt)
    input_csv = "/home_cerberus/disk3/larissa.gomide/PKDD/dataset_all_f.csv"
    df = pd.read_csv(input_csv) 

    # ============================
    # Execução com o modelo SMALL (Janus-Pro-1B)
    # ============================
    print("Executando modelo SMALL (Janus-Pro-1B)...")
    small_model_path = "deepseek-ai/Janus-Pro-1B"
    small_processor = VLChatProcessor.from_pretrained(small_model_path)
    # Para token-based usamos image_start_tag e formato padrão (use_alternate=False)
    small_model = AutoModelForCausalLM.from_pretrained(small_model_path, trust_remote_code=True)
    small_model = small_model.to(torch.bfloat16).cuda().eval()
    
    # Diretório para salvar as imagens geradas pelo modelo pequeno
    output_dir_positive = "/home_cerberus/disk3/larissa.gomide/oficial/generated_oficial/positive"
    output_dir_very_positive = "/home_cerberus/disk3/larissa.gomide/oficial/generated_oficial/very_positive"
    output_dir_negative = "/home_cerberus/disk3/larissa.gomide/oficial/generated_oficial/negative"
    output_dir_very_negative = "/home_cerberus/disk3/larissa.gomide/oficial/generated_oficial/very_negative"

    os.makedirs(output_dir_positive, exist_ok=True)
    os.makedirs(output_dir_very_positive, exist_ok=True)
    os.makedirs(output_dir_negative, exist_ok=True)
    os.makedirs(output_dir_very_negative, exist_ok=True)
    
    generated_filenames_positive = []
    generated_filenames_very_positive = []
    generated_filenames_negative = []
    generated_filenames_very_negative = []

    start_time_small = time.time()
    for idx, row in df.iterrows():
        prompt_positive = row["Description_positive"]
        prompt_very_positive = row["Description_very_positive"]
        prompt_negative = row["Description_negative"]
        prompt_very_negative = row["Description_very_negative"]

        prompt_positive = get_prompt(small_processor, prompt_positive, is_token_based=True, use_alternate=False)
        prompt_very_positive = get_prompt(small_processor, prompt_very_positive, is_token_based=True, use_alternate=False)
        prompt_negative = get_prompt(small_processor, prompt_negative, is_token_based=True, use_alternate=False)
        prompt_very_negative = get_prompt(small_processor, prompt_very_negative, is_token_based=True, use_alternate=False)

        # Define o caminho de saída da imagem (nome pode ser ajustado conforme necessidade)
        out_filename_positive = os.path.join(output_dir_positive, f"img_small_{idx}.jpg")
        output_filename_very_positive = os.path.join(output_dir_very_positive, f"img_small_{idx}.jpg")
        out_filename_negative = os.path.join(output_dir_negative, f"img_small_{idx}.jpg")
        out_filename_very_negative = os.path.join(output_dir_very_negative, f"img_small_{idx}.jpg")

        generate_token_based(small_model, small_processor, prompt_positive, parallel_size=1, save_path=out_filename_positive)
        generate_token_based(small_model, small_processor, prompt_very_positive, parallel_size=1, save_path=output_filename_very_positive)
        generate_token_based(small_model, small_processor, prompt_negative, parallel_size=1, save_path=out_filename_negative)
        generate_token_based(small_model, small_processor, prompt_very_negative, parallel_size=1, save_path=out_filename_very_negative)
        
        generated_filenames_positive.append(out_filename_positive)
        generated_filenames_very_positive.append(output_filename_very_positive)
        generated_filenames_negative.append(out_filename_negative)
        generated_filenames_very_negative.append(out_filename_very_negative)

        print(f"[SMALL] Processado prompt {idx}")
    end_time_small = time.time()
    print(f"Tempo total de geração (SMALL): {end_time_small - start_time_small:.2f} segundos")
    
    # Cria uma nova tabela com a coluna 'generated_filename'
    df_positive = df.copy()
    df_very_positive = df.copy()
    df_negative = df.copy()
    df_very_negative = df.copy()

    df_positive['generated_filename'] = generated_filenames_positive
    df_very_positive['generated_filename'] = generated_filenames_very_positive
    df_negative['generated_filename'] = generated_filenames_negative
    df_very_negative['generated_filename'] = generated_filenames_very_negative

    df_positive.to_csv("/home_cerberus/disk3/larissa.gomide/oficial/positive.csv", index=False)
    df_very_positive.to_csv("/home_cerberus/disk3/larissa.gomide/oficial/very_positive.csv", index=False)
    df_negative.to_csv("/home_cerberus/disk3/larissa.gomide/oficial/negative.csv", index=False)
    df_very_negative.to_csv("/home_cerberus/disk3/larissa.gomide/oficial/very_negative.csv", index=False)

    # Simulação de envio (print antes de enviar)
    print("De:", seu_email)
    print("Para:", destinatario)
    print("Assunto:", msg["Subject"])
    print("Corpo:\n", corpo)

    # Descomente a linha abaixo para enviar o e-mail após testar
    try:
         servidor = smtplib.SMTP("smtp.gmail.com", 587)
         servidor.starttls()
         servidor.login(seu_email, senha)
         servidor.sendmail(seu_email, destinatario, msg.as_string())
         servidor.quit()
         print("E-mail enviado com sucesso!")
    except Exception as e:
         print(f"Erro ao enviar e-mail: {e}")
    
if __name__ == "__main__":
    main()
