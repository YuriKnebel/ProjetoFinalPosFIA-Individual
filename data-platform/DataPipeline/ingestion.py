import os
import json
import pandas as pd
from utils import save_dataframe_to_postgres

def run_csv_ingestion(table_name: str, conn_id: str, pasta_origem: str, config_file: str):
    """Executa a ingestão de um único arquivo CSV mapeado no escopo do JSON."""
    
    # 1. Validação de Escopo via JSON
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado em: {config_file}")
        
    with open(config_file, "r") as f:
        config = json.load(f)
        
    tabelas_permitidas = config.get("ingestion_table", {}).get("using_csv", [])
    
    if table_name not in tabelas_permitidas:
        raise ValueError(f"A tabela '{table_name}' NÃO está homologada no escopo do JSON: {tabelas_permitidas}")
        
    if not os.path.exists(pasta_origem):
        raise FileNotFoundError(f"A pasta de origem {pasta_origem} não existe.")

    # 2. Localização do arquivo correspondente na pasta
    # Buscamos arquivos que, normalizados, batam com o nome da tabela solicitado
    arquivo_alvo = None
    for arquivo in os.listdir(pasta_origem):
        nome_arq, extensao = os.path.splitext(arquivo)
        if nome_arq.lower().replace("-", "_").replace(" ", "_") == table_name:
            arquivo_alvo = arquivo
            break
            
    if not arquivo_alvo:
        raise FileNotFoundError(f"Nenhum arquivo correspondente à tabela '{table_name}' foi encontrado em {pasta_origem}")
        
    caminho_completo = os.path.join(pasta_origem, arquivo_alvo)
    _, extensao = os.path.splitext(arquivo_alvo)

    print(f"📖 Lendo {arquivo_alvo} para ingestão da tabela '{table_name}'...")
    
    # 3. Leitura resiliente do arquivo
    if extensao.lower() == '.csv':
        try:
            df = pd.read_csv(caminho_completo, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(caminho_completo, encoding='latin-1')
    elif extensao.lower() == '.json':
        df = pd.read_json(caminho_completo)
    elif extensao.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(caminho_completo)
    else:
        raise ValueError(f"Formato {extensao} não suportado para a tabela {table_name}")

    # 4. Gravação de alta performance via utils
    save_dataframe_to_postgres(df=df, table_name=table_name, conn_id=conn_id)
    print(f"Ingestão da tabela '{table_name}' concluída com sucesso!")