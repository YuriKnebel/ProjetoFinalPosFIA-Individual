import io
import pandas as pd
import os

def get_database_connection(conn_id: str = "postgres_data_db"):
    """Retorna uma conexão ativa com o banco.

    Detecta automaticamente se está rodando dentro do fluxo do Airflow (usa
    PostgresHook) ou de forma isolada (usa SQLAlchemy).
    """
    # Se a variável de ambiente do Airflow existir, usamos o Hook Nativo
    if "AIRFLOW_HOME" in os.environ:
        try:
            from airflow.providers.postgres.hooks.postgres import PostgresHook

            print(
                f"🔗 [CONEXÃO] Ambiente Airflow detectado. Usando PostgresHook('{conn_id}')."
            )
            pg_hook = PostgresHook(postgres_conn_id=conn_id)
            return pg_hook.get_conn()
        except ImportError:
            print(
                "[CONEXÃO] Aviso: AIRFLOW_HOME ativa, mas falha ao importar PostgresHook. Tentando fallback para SQLAlchemy..."
            )

    # Fallback para execução local/notebook via SQLAlchemy
    from sqlalchemy import create_engine

    # Mapeando o host baseado no ambiente (se roda dentro do ecossistema docker ou na máquina local)
    # No docker o host do banco chama-se 'postgres'. Na máquina local acessamos via 'localhost'
    host = "postgres" if os.path.exists("/.dockerenv") else "localhost"
    conn_str = f"postgresql://airflow:airflow@{host}:5432/data"

    print(
        f"🔗 [CONEXÃO] Execução isolada detectada (Local/Notebook). Conectando via SQLAlchemy em '{host}'."
    )
    engine = create_engine(conn_str)

    # Retorna uma conexão raw estável para o Pandas read_sql_query não reclamar
    return engine.raw_connection()

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

def append_dataframe_to_postgres(
    df: pd.DataFrame, table_name: str, conn_id: str
):
    """Insere os dados de um chunk em uma tabela já existente usando a conexão híbrida."""
    conn = get_database_connection(conn_id)
    cursor = conn.cursor()

    try:
        output = io.StringIO()
        df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        cursor.copy_expert(
            f'COPY "{table_name}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'',
            output,
        )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(
            f"Falha no append do chunk na tabela {table_name}: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()

def save_dataframe_to_postgres(df: pd.DataFrame, table_name: str, conn_id: str):
    """Cria/Recria a tabela e insere os dados de forma ultra rápida usando a conexão híbrida."""
    # Busca a conexão inteligente
    conn = get_database_connection(conn_id)
    cursor = conn.cursor()

    try:
        colunas_sql = map_pandas_to_postgres_types(df)

        print(f"♻️ Reiniciando tabela '{table_name}' no banco...")
        cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE;')

        sql_create = f'CREATE TABLE "{table_name}" ({", ".join(colunas_sql)});'
        cursor.execute(sql_create)

        output = io.StringIO()
        df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        print(f"📥 Despejando {len(df)} linhas em '{table_name}' via COPY...")
        cursor.copy_expert(
            f'COPY "{table_name}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'',
            output,
        )

        conn.commit()
        print(f"✅ Tabela '{table_name}' populada com sucesso!")

    except Exception as e:
        conn.rollback()
        raise RuntimeError(
            f"Falha crítica na gravação da tabela {table_name}: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()