import io
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook

def map_pandas_to_postgres_types(df: pd.DataFrame) -> list:
    """Mapeia os dtypes do Pandas para tipos de dados compatíveis com o PostgreSQL."""
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
    return colunas

def append_dataframe_to_postgres(df: pd.DataFrame, table_name: str, conn_id: str):
    """Insere os dados de um chunk em uma tabela já existente via COPY."""
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    try:
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        
        cursor.copy_expert(f'COPY "{table_name}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Falha no append do chunk na tabela {table_name}: {str(e)}")
    finally:
        cursor.close()
        conn.close()

def save_dataframe_to_postgres(df: pd.DataFrame, table_name: str, conn_id: str):
    """Cria/Recria a tabela e insere os dados de forma ultra rápida via copy_expert."""
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    try:
        # Gera a estrutura de colunas dinamicamente
        colunas_sql = map_pandas_to_postgres_types(df)
        
        print(f"Reiniciando tabela '{table_name}' no banco...")
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')
        
        sql_create = f'CREATE TABLE "{table_name}" ({", ".join(colunas_sql)});'
        cursor.execute(sql_create)
        
        # Criação do buffer em memória RAM para o COPY
        output = io.StringIO()
        df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        
        print(f"Despejando {len(df)} linhas em '{table_name}' via COPY...")
        cursor.copy_expert(f'COPY "{table_name}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)
        
        conn.commit()
        print(f"Tabela '{table_name}' populada com sucesso!")
        
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Falha crítica na gravação da tabela {table_name}: {str(e)}")
        
    finally:
        cursor.close()
        conn.close()