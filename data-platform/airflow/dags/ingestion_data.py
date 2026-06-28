from datetime import datetime
import os
import pandas as pd
from airflow import DAG
from airflow.decorators import task
from airflow.providers.postgres.hooks.postgres import PostgresHook

# Constantes centralizadas para fácil manutenção futura
CONN_ID = "postgres_data_db"  # O ID da conexão que foi criada na UI do Airflow via Docker Compose
PASTA_DATA = "/opt/airflow/data/csv"

with DAG(
    dag_id="loadfile_csv_to_postgres",
    start_date=datetime(2026, 6, 24),
    schedule=None,
    catchup=False,
    tags=["ingestion", "postgres"],
) as dag:

    @task
    def loadfile(conn_id: str, pasta_origem: str):
        import io
        
        pg_hook = PostgresHook(postgres_conn_id=conn_id)
        conn_bruta = pg_hook.get_conn()
        cursor = conn_bruta.cursor()
        
        if not os.path.exists(pasta_origem):
            raise FileNotFoundError(f"A pasta {pasta_origem} não existe no container.")
            
        arquivos = os.listdir(pasta_origem)
        print(f"Arquivos detectados para processamento: {arquivos}")
        
        if not arquivos:
            print("Nenhum arquivo encontrado para processar.")
            return

        for arquivo in arquivos:
            caminho_completo = os.path.join(pasta_origem, arquivo)
            
            if os.path.isdir(caminho_completo):
                continue
                
            nome_tabela, extensao = os.path.splitext(arquivo)
            nome_tabela = nome_tabela.lower().replace("-", "_").replace(" ", "_")
            
            try:
                # 1. Tratamento robusto de leitura com fallback de encoding para CSVs
                if extensao.lower() == '.csv':
                    try:
                        df = pd.read_csv(caminho_completo, encoding='utf-8')
                    except UnicodeDecodeError:
                        print(f"Aviso: UTF-8 falhou para {arquivo}. Tentando ler com latin-1...")
                        df = pd.read_csv(caminho_completo, encoding='latin-1')
                elif extensao.lower() == '.json':
                    df = pd.read_json(caminho_completo)
                elif extensao.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(caminho_completo)
                else:
                    print(f"Formato '{extensao}' ignorado para o arquivo: {arquivo}")
                    continue
                
                print(f"Iniciando carga de {arquivo} para tabela '{nome_tabela}'...")
                
                # 2. Mapeamento de colunas
                colunas = []
                for col, dtype in zip(df.columns, df.dtypes):
                    col_nome = str(col).lower().replace("-", "_").replace(" ", "_").replace(".", "_")
                    if "int" in str(dtype):
                        pg_type = "BIGINT"
                    elif "float" in str(dtype):
                        pg_type = "DOUBLE PRECISION"
                    elif "bool" in str(dtype):
                        pg_type = "BOOLEAN"
                    elif "datetime" in str(dtype):
                        pg_type = "TIMESTAMP"
                    else:
                        pg_type = "TEXT"
                    colunas.append(f'"{col_nome}" {pg_type}')
                
                # --- AQUI ESTÁ O DROP E CREATE QUE VOCÊ MENCIONOU ---
                # Garante que a tabela antiga seja totalmente excluída antes de iniciar a nova carga
                cursor.execute(f'DROP TABLE IF EXISTS "{nome_tabela}" CASCADE;')
                sql_create = f'CREATE TABLE "{nome_tabela}" ({", ".join(colunas)});'
                cursor.execute(sql_create)
                
                # 3. Carga Ultra Rápida via cópia em memória RAM
                output = io.StringIO()
                df.to_csv(output, sep='\t', header=False, index=False)
                output.seek(0)
                
                cursor.copy_expert(f'COPY "{nome_tabela}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)
                conn_bruta.commit()
                
                print(f"Sucesso! Tabela '{nome_tabela}' criada/recriada e populada com {len(df)} linhas.")
                
            except Exception as e:
                conn_bruta.rollback()
                print(f"Falha ao processar o arquivo {arquivo}. Erro: {str(e)}")
                print("Aviso: Pulando para o próximo arquivo para não travar o pipeline...")
                continue # Permite que a DAG continue lendo os arquivos seguintes mesmo se um falhar
        
        cursor.close()
        conn_bruta.close()

    # Executa a task passando os parâmetros configurados
    loadfile(conn_id=CONN_ID, pasta_origem=PASTA_DATA)
