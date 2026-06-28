import os
import json
import io
import numpy as np
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)

def sanitize_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df_clean = df.copy()
    params = config["cleaning_parameters"]

    if "days_employed" in df_clean.columns:
        df_clean["days_employed_anom"] = (df_clean["days_employed"] == params["days_employed_anomaly"]).astype(int)
        df_clean["days_employed"] = df_clean["days_employed"].replace(params["days_employed_anomaly"], np.nan)

    for col in params["columns_to_absolute"]:
        if col in df_clean.columns:
            df_clean[col] = np.abs(df_clean[col])

    for col in ["days_employed", "days_birth", "days_registration", "days_id_publish"]:
        if col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype("Int64")
            except:
                df_clean[col] = np.round(df_clean[col])
                df_clean[col] = (df_clean[col].dt.days if hasattr(df_clean[col], "dt") else df_clean[col])  
                df_clean[col] = (df_clean[col].apply(lambda x: str(int(x)) if pd.notnull(x) else ""))
    
    string_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in string_cols:
        if col not in ["days_employed","days_birth","days_registration","days_id_publish"]:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def create_clean_table_schema(cursor, source_table: str, target_table: str):
    cursor.execute(f'DROP TABLE IF EXISTS "{target_table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{target_table}" (LIKE "{source_table}" INCLUDING ALL);')
    cursor.execute(f'ALTER TABLE "{target_table}" ADD COLUMN IF NOT EXISTS "days_employed_anom" INTEGER;')

def run_sanitization(conn_id: str):
    """Função mestre que será chamada nativamente pela Task do Airflow."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config_pipeline.json")
    config = load_config(config_path)

    # O PostgresHook agora roda dentro do contexto correto e vai achar o banco na hora!
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_table"]
    output_table = config["database"]["output_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]

    print(f"Preparando tabela de destino '{output_table}'...")
    create_clean_table_schema(cursor, input_table, output_table)
    conn.commit()

    offset = 0
    print(f"Iniciando processamento em lotes nativo...")

    while True:
        query = f'SELECT * FROM "{input_table}" LIMIT {chunk_size} OFFSET {offset};'
        chunk_df = pd.read_sql(query, conn)

        if chunk_df.empty:
            break

        print(f"Processando lote (Offset: {offset}, Linhas: {len(chunk_df)})...")
        cleaned_df = sanitize_data(chunk_df, config)

        output = io.StringIO()
        cleaned_df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        cursor.copy_expert(
            f'COPY "{output_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
        )
        conn.commit()

        offset += chunk_size

    cursor.close()
    conn.close()
    print("--- Pipeline de sanitização finalizado com sucesso! ---")


def sanitize_prev_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    # 1. Padroniza colunas de texto cruciais
    if "name_contract_status" in df_clean.columns:
        df_clean["name_contract_status"] = df_clean["name_contract_status"].astype(str).str.strip().str.title()
        
    # 2. Trata valores nulos ou negativos no valor pedido (amt_application)
    if "amt_application" in df_clean.columns:
        df_clean["amt_application"] = df_clean["amt_application"].fillna(0)
        df_clean["amt_application"] = np.where(df_clean["amt_application"] < 0, 0, df_clean["amt_application"])
        
    return df_clean

def run_prev_sanitization(conn_id: str):
    """Função mestre para limpar a tabela previous_application em lotes."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config_pipeline.json")
    config = load_config(config_path)

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["input_prev_table"]
    output_table = config["database"]["output_prev_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]

    print(f"Preparando tabela de destino histórica '{output_table}'...")
    cursor.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE;')
    cursor.execute(f'CREATE TABLE "{output_table}" (LIKE "{input_table}" INCLUDING ALL);')
    conn.commit()

    offset = 0
    print("Iniciando sanitização do histórico em lotes...")

    while True:
        query = f'SELECT * FROM "{input_table}" LIMIT {chunk_size} OFFSET {offset};'
        chunk_df = pd.read_sql(query, conn)

        if chunk_df.empty:
            break

        cleaned_df = sanitize_prev_data(chunk_df)

        output = io.StringIO()
        cleaned_df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        cursor.copy_expert(f'COPY "{output_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)
        conn.commit()
        offset += chunk_size

    cursor.close()
    conn.close()
    print(f"--- Histórico limpo com sucesso na tabela '{output_table}' ---")
