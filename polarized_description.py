import smtplib
import time
import pandas as pd
import requests
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def rewrite_description(description, sentiment):
    url = "https://text.pollinations.ai/"
    headers = {'Content-Type': 'application/json'}
    payload = {
        'messages': [
            {'role': 'user', 'content': f"Rewrite the following text to have a tone that is {sentiment}:\n\n{description}. Be faithful to the original text. It must have the same amount of tokens."}
        ],
        'seed': 42,
        'model': 'mistral'
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Erro ao chamar API: {response.status_code}")
        return None

def process_csv(csv_file):
    df = pd.read_csv(csv_file)
    print("Carregou a base de dados")
    
    df_positive = pd.DataFrame(columns=["description_original", "description_positive"])
    df_very_positive = pd.DataFrame(columns=["description_original", "description_very_positive"])
    df_negative = pd.DataFrame(columns=["description_original", "description_negative"])
    df_very_negative = pd.DataFrame(columns=["description_original", "description_very_negative"])
    
    start_time = time.time()
    for i, row in df.iterrows():
        print(f"Processando linha {i+1} de {len(df)}")
        original_text = row["Description"]
        
        positive_text = rewrite_description(original_text, "positive")
        very_positive_text = rewrite_description(original_text, "very positive")
        negative_text = rewrite_description(original_text, "negative")
        very_negative_text = rewrite_description(original_text, "very negative")
        
        df_positive.loc[i] = [original_text, positive_text]
        df_very_positive.loc[i] = [original_text, very_positive_text]
        df_negative.loc[i] = [original_text, negative_text]
        df_very_negative.loc[i] = [original_text, very_negative_text]
    
    end_time = time.time()
    print(f"Tempo total gasto: {end_time - start_time:.2f} segundos")
    
    df_positive.to_csv("dataset_positive_f.csv", index=False)
    df_very_positive.to_csv("dataset_very_positive_f.csv", index=False)
    df_negative.to_csv("dataset_negative_f.csv", index=False)
    df_very_negative.to_csv("dataset_very_negative_f.csv", index=False)
    print("Arquivos salvos com sucesso")

def send_email():
    seu_email = "codigosecreto2025@gmail.com"
    senha = "bpij xclh mwqr sofe"
    destinatario = "laladg18@gmail.com"
    
    msg = MIMEMultipart()
    msg["From"] = seu_email
    msg["To"] = destinatario
    msg["Subject"] = "Fim execução polarized_description.py"
    msg.attach(MIMEText("Fim do script Python", "plain"))
    
    try:
        servidor = smtplib.SMTP("smtp.gmail.com", 587)
        servidor.starttls()
        servidor.login(seu_email, senha)
        servidor.sendmail(seu_email, destinatario, msg.as_string())
        servidor.quit()
        print("E-mail enviado com sucesso!")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

#Processa o df_sample 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa um CSV para gerar descrições polarizadas e envia e-mail ao final.")
    parser.add_argument("csv_file", help="Caminho para o arquivo CSV de entrada")
    args = parser.parse_args()
    
    process_csv(args.csv_file)
    send_email()
