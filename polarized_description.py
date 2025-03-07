import time
import requests
import pandas as pd

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

print("Começo do script")
# Configurações do e-mail
seu_email = "codigosecreto2025@gmail.com"
senha = "bpij xclh mwqr sofe"
destinatario = "laladg18@gmail.com"

# Criando a mensagem
msg = MIMEMultipart()
msg["From"] = seu_email
msg["To"] = destinatario
msg["Subject"] = "Fim execução get_scores.py"

corpo = "Fim do script Python"
msg.attach(MIMEText(corpo, "plain"))

# Função para chamar a API com retry e backoff exponencial
def rewrite_description(description, sentiment, max_retries=5, base_delay=2):
    url = "https://text.pollinations.ai/"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'messages': [
            {'role': 'user', 'content': f"Rewrite the following text to have a tone that is {sentiment}:\n\n{description}. Be faithful to the original text. It must have the same amount of tokens."}
        ],
        'seed': 42,
        'model': 'mistral'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                return response.text
            
            print(f"Tentativa {attempt+1}/{max_retries} falhou com erro {response.status_code}. Tentando novamente...")
            time.sleep(base_delay * (2 ** attempt))  # Atraso exponencial: 2s, 4s, 8s, 16s...
        except requests.RequestException as e:
            print(f"Erro de conexão: {e}. Tentando novamente...")
            time.sleep(base_delay * (2 ** attempt))
    
    print("Erro persistente. Salvando entrada para tentar novamente mais tarde.")
    return None

# Carregar os datasets ou criar novos caso não existam
try:
    df_positive = pd.read_csv("dataset_positive_f.csv")
    df_very_positive = pd.read_csv("dataset_very_positive_f.csv")
    df_negative = pd.read_csv("dataset_negative_f.csv")
    df_very_negative = pd.read_csv("dataset_very_negative_f.csv")
except FileNotFoundError:
    df_positive = pd.DataFrame(columns=["description_original", "description_positive"])
    df_very_positive = pd.DataFrame(columns=["description_original", "description_very_positive"])
    df_negative = pd.DataFrame(columns=["description_original", "description_negative"])
    df_very_negative = pd.DataFrame(columns=["description_original", "description_very_negative"])

# Carregar dataset de entrada
df = pd.read_csv("/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_SMALL_with_gen_scored.csv")

# Listas para armazenar as falhas
failed_entries = []

# Loop para processar os textos
for i, row in df.iterrows():
    print(f"Processando linha {i+1} de {len(df)}")
    original_text = row["Description"]
    
    results = {
        "positive": rewrite_description(original_text, "positive"),
        "very positive": rewrite_description(original_text, "very positive"),
        "negative": rewrite_description(original_text, "negative"),
        "very negative": rewrite_description(original_text, "very negative")
    }
    
    # Salvar falhas para tentar depois
    for sentiment, text in results.items():
        if text is None:
            failed_entries.append((original_text, sentiment))
    
    # Adicionar ao DataFrame apenas se a requisição foi bem-sucedida
    if results["positive"]:
        df_positive.loc[i] = [original_text, results["positive"]]
    if results["very positive"]:
        df_very_positive.loc[i] = [original_text, results["very positive"]]
    if results["negative"]:
        df_negative.loc[i] = [original_text, results["negative"]]
    if results["very negative"]:
        df_very_negative.loc[i] = [original_text, results["very negative"]]
    
    # Salvar progresso periodicamente
    if i % 10 == 0:
        df_positive.to_csv("dataset_positive_f.csv", index=False)
        df_very_positive.to_csv("dataset_very_positive_f.csv", index=False)
        df_negative.to_csv("dataset_negative_f.csv", index=False)
        df_very_negative.to_csv("dataset_very_negative_f.csv", index=False)
        
    time.sleep(1)  # Pequeno atraso para evitar sobrecarga da API

# Tentar regerar as falhas ao final
print("Tentando regerar descrições que falharam...")
recovered_entries = []
for original_text, sentiment in failed_entries:
    regenerated_text = rewrite_description(original_text, sentiment)
    if regenerated_text:
        recovered_entries.append((original_text, sentiment, regenerated_text))
    time.sleep(1)

# Salvar as entradas recuperadas
for original_text, sentiment, regenerated_text in recovered_entries:
    if sentiment == "positive":
        df_positive.loc[len(df_positive)] = [original_text, regenerated_text]
    elif sentiment == "very positive":
        df_very_positive.loc[len(df_very_positive)] = [original_text, regenerated_text]
    elif sentiment == "negative":
        df_negative.loc[len(df_negative)] = [original_text, regenerated_text]
    elif sentiment == "very negative":
        df_very_negative.loc[len(df_very_negative)] = [original_text, regenerated_text]

# Salvar datasets finais
df_positive.to_csv("dataset_positive_f.csv", index=False)
df_very_positive.to_csv("dataset_very_positive_f.csv", index=False)
df_negative.to_csv("dataset_negative_f.csv", index=False)
df_very_negative.to_csv("dataset_very_negative_f.csv", index=False)

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


print("Processo concluído!")
