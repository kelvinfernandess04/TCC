import requests

URL_BASE = "http://136.248.91.108:8000" 

def validar_conexao():
    print("--- 1. Validando Conexão com o Banco ---")
    try:
        response = requests.get(f"{URL_BASE}/health", timeout=5)
        print(f"Resposta Health: {response.json()}")
    except Exception as e:
        print(f"Erro ao conectar na API: {e}")

def testar_batch():
    print("\n--- 2. Testando busca de IDs (Pode vir vazio se o banco estiver limpo) ---")
    try:
        params = {"ids": "1,2,3"} 
        response = requests.get(f"{URL_BASE}/signatures/batch", params=params)
        print(f"Status: {response.status_code}")
        print(f"Dados: {response.json()}")
    except Exception as e:
        print(f"Falha na requisição batch: {e}")

if __name__ == "__main__":
    validar_conexao()
    testar_batch()