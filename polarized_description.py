import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import pandas as pd
from transformers import pipeline

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
df = pd.read_csv("/home_cerberus/disk3/larissa.gomide/oficial/amostraGauss/sampled_SMALL_with_gen_scored.csv")
print("Carregou a base de dados")

# Inicializa o pipeline com DeepSeek-V3
pipe = pipeline("text-generation", model="deepseek-ai/DeepSeek-V2", trust_remote_code=True)
print("Carregou o modelo DeepSeek-V2")

# Função para reescrever frases com um viés específico (positivo ou negativo)
def rewrite_description(description, sentiment):
    # Estrutura de mensagens (role: user)
    messages = [
        {"role": "user", "content": f"Rewrite the following text to have a tone that is {sentiment}:\n\n{description}"}
    ]
    # Gera a resposta com o modelo
    output = pipe(messages)
    return output[0]['generated_text'].strip()

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
df_positive.to_csv("/home_cerberus/disk3/larissa.gomide/PKDD/dataset_positive.csv", index=False)
df_very_positive.to_csv("/home_cerberus/disk3/larissa.gomide/PKDD/dataset_very_positive.csv", index=False)
df_negative.to_csv("/home_cerberus/disk3/larissa.gomide/PKDD/dataset_negative.csv", index=False)
df_very_negative.to_csv("/home_cerberus/disk3/larissa.gomide/PKDD/dataset_very_negative.csv", index=False)
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
