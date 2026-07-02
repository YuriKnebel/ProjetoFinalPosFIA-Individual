import numpy as np
import pandas as pd
from utils import save_dataframe_to_postgres, append_dataframe_to_postgres, get_database_connection

def load_data_from_abt(conn_id: str, abt_table_name: str) -> pd.DataFrame:
    """Busca a ABT usando a nossa conexão inteligente camaleônica."""
    # 🌟 MÁGICA AQUI: Ele descobre sozinho se usa Hook ou SQLAlchemy
    conn = get_database_connection(conn_id)

    query = f'SELECT * FROM "{abt_table_name}";'
    df = pd.read_sql_query(query, conn)

    conn.close()

    return df

def sanitize_data_chunk(df: pd.DataFrame, anomaly_value: int, cols_to_absolute: list) -> pd.DataFrame:
    """Aplica a sua lógica exata de higienização em cada chunk de dados."""
    df_clean = df.copy()

    # 1. Tratamento da anomalia de dias trabalhados
    if "days_employed" in df_clean.columns:
        df_clean["days_employed_anom"] = (df_clean["days_employed"] == anomaly_value).astype(int)
        df_clean["days_employed"] = df_clean["days_employed"].replace(anomaly_value, np.nan)

    # 2. Transformação em valores absolutos
    for col in cols_to_absolute:
        if col in df_clean.columns:
            df_clean[col] = np.abs(df_clean[col])

    # 3. Tratamento robusto de tipagem e arredondamento das colunas temporais
    target_cols = ["days_employed", "days_birth", "days_registration", "days_id_publish"]
    for col in target_cols:
        if col in df_clean.columns:
            try:
                df_clean[col] = df_clean[col].astype("Int64")
            except:
                df_clean[col] = np.round(df_clean[col])
                df_clean[col] = (df_clean[col].dt.days if hasattr(df_clean[col], "dt") else df_clean[col])  
                df_clean[col] = (df_clean[col].apply(lambda x: str(int(x)) if pd.notnull(x) else ""))
    
    # 4. Limpeza de strings (strip) evitando colunas numéricas tratadas acima
    string_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in string_cols:
        if col not in target_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    return df_clean

def run_sanitization(conn_id, input_table, output_table, anomaly_value, cols_to_absolute, chunk_size):
    """Higieniza a tabela application_train aplicando sua lógica em chunks."""
    print(f"Sanitizando {input_table} -> {output_table} em blocos de {chunk_size}...")
    conn = get_database_connection(conn_id) 
    
    query = f'SELECT * FROM "{input_table}";'
    chunks = pd.read_sql_query(query, conn, chunksize=chunk_size)
    
    first_chunk = True
    for df_chunk in chunks:
        # Executa a sua função de higienização completa no bloco atual
        df_chunk_clean = sanitize_data_chunk(df_chunk, anomaly_value, cols_to_absolute)
        
        if first_chunk:
            save_dataframe_to_postgres(df_chunk_clean, output_table, conn_id)
            first_chunk = False
        else:
            append_dataframe_to_postgres(df_chunk_clean, output_table, conn_id)
            
    conn.close()
    print(f"Sanitização completa de {output_table} finalizada!")


def run_prev_sanitization(conn_id, input_table, output_table, chunk_size):
    """Higieniza a tabela previous_application em chunks."""
    print(f"Sanitizando Histórico: {input_table} -> {output_table} em blocos de {chunk_size}...")
    conn = get_database_connection(conn_id) 
    
    query = f'SELECT * FROM "{input_table}";'
    chunks = pd.read_sql_query(query, conn, chunksize=chunk_size)
    
    first_chunk = True
    for df_chunk in chunks:
        # Padronização básica de strings para a tabela de histórico
        if "name_contract_status" in df_chunk.columns:
            df_chunk["name_contract_status"] = df_chunk["name_contract_status"].astype(str).str.strip()
            
        if first_chunk:
            save_dataframe_to_postgres(df_chunk, output_table, conn_id)
            first_chunk = False
        else:
            append_dataframe_to_postgres(df_chunk, output_table, conn_id)
            
    conn.close()
    print(f"Sanitização completa de {output_table} finalizada!")