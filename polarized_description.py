import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import pandas as pd
from transformers import pipeline
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

print("Começo do script")

# Configurações do e-mail
seu_email = "codigosecreto2025@gmail.com"
senha = "bpij xclh mwqr sofe"
destinatario = "laladg18@gmail.com"

# Criando a mensagem
msg = MIMEMultipart()
msg["From"] = seu_email
msg["To"] = destinatario
msg["Subject"] = "Fim execução polarized_description.py"

corpo = "Fim do script Python"
msg.attach(MIMEText(corpo, "plain"))


# Carrega o dataset original (ajuste o nome do arquivo se necessário)
df = df_small
print("Carregou a base de dados")

# Função para reescrever frases com um viés específico (positivo ou negativo)
def rewrite_description(description, sentiment):
    url = "https://text.pollinations.ai/"  # URL da API Pollinations
    headers = {
        'Content-Type': 'application/json',
    }
    payload = {
        'messages': [
            {'role': 'user', 'content': f"Rewrite the following text to have a tone that is {sentiment}:\n\n{description}.Be faityfull to the original text."}
        ],
        'seed': 42,
        'model': 'mistral'  # Definindo o modelo (caso precise mudar, altere conforme o necessário)
    }
    
    # Faz a requisição à API Pollinations
    response = requests.post(url, headers=headers, json=payload)

    print(f"Status Code: {response.status_code}")
    print(f"Response Text: {response.text}")
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Erro ao chamar API: {response.status_code}")
        return None


# Criar DataFrames separados
df_positive = pd.DataFrame(columns=["description_original", "description_positive"])
df_very_positive = pd.DataFrame(columns=["description_original", "description_very_positive"])
df_negative = pd.DataFrame(columns=["description_original", "description_negative"])
df_very_negative = pd.DataFrame(columns=["description_original", "description_very_negative"])

# Inicia a contagem de tempo
print("Inicia a contagem de tempo")
start_time = time.time()

# Processa cada linha do dataset
for i, row in df.iterrows():
    print(f"Row:{i} of {len(df)}")
    original_text = row["Description"]
    
    # Gera a versão positiva e negativa
    positive_text = rewrite_description(original_text, "positive")
    very_positive_text = rewrite_description(original_text, "very positive")
    negative_text = rewrite_description(original_text, "negative")
    very_negative_text = rewrite_description(original_text, "very negative")

    # Adiciona ao DataFrame correspondente
    df_positive.loc[i] = [original_text, positive_text]
    df_very_positive.loc[i] = [original_text, very_positive_text]
    df_negative.loc[i] = [original_text, negative_text]
    df_very_negative.loc[i] = [original_text, very_negative_text]

# Tempo total de processamento
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tempo total gasto: {elapsed_time:.2f} segundos")

# Salva os datasets separados
df_positive.to_csv("dataset_positive_f.csv", index=False)
df_very_positive.to_csv("dataset_very_positive_f.csv", index=False)
df_negative.to_csv("dataset_negative_f.csv", index=False)
df_very_negative.to_csv("dataset_very_negative_f.csv", index=False)
print("Arquivos salvos com sucesso")


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
