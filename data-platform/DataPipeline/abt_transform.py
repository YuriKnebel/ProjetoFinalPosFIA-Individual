import os
import json
import io
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gera novas variáveis explicativas (Engenharia de Features) para a ABT."""
    df_features = df.copy()

    # Evitando divisão por zero
    if "amt_income_total" in df_features.columns:
        df_features["amt_income_total"] = df_features["amt_income_total"].replace(0, 0.001)

    if "amt_credit" in df_features.columns and "amt_income_total" in df_features.columns:
        df_features["fe_credit_income_percent"] = (df_features["amt_credit"] / df_features["amt_income_total"])

    if "amt_annuity" in df_features.columns and "amt_income_total" in df_features.columns:
        df_features["fe_annuity_income_percent"] = (df_features["amt_annuity"] / df_features["amt_income_total"])

    if "cnt_fam_members" in df_features.columns and "amt_income_total" in df_features.columns:
        df_features["cnt_fam_members"] = df_features["cnt_fam_members"].fillna(1).replace(0, 1)
        df_features["fe_income_per_person"] = (df_features["amt_income_total"] / df_features["cnt_fam_members"])

    return df_features


def create_abt_table_schema(cursor, sample_df: pd.DataFrame, target_table: str):
    """Cria a estrutura da tabela ABT dinamicamente baseada nas colunas do DataFrame."""
    colunas = []
    for col, dtype in zip(sample_df.columns, sample_df.dtypes):
        if "int" in str(dtype).lower():
            pg_type = "BIGINT"
        elif "float" in str(dtype).lower():
            pg_type = "DOUBLE PRECISION"
        elif "bool" in str(dtype).lower():
            pg_type = "BOOLEAN"
        else:
            pg_type = "TEXT"
        colunas.append(f'"{col}" {pg_type}')

    cursor.execute(f'DROP TABLE IF EXISTS "{target_table}" CASCADE;')
    sql_create = f'CREATE TABLE "{target_table}" ({", ".join(colunas)});'
    cursor.execute(sql_create)


def run_abt_generation(conn_id: str):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config_pipeline.json")
    config = load_config(config_path)

    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    input_table = config["database"]["output_table"]
    output_table = config["database"]["abt_table"]
    chunk_size = config["cleaning_parameters"]["chunk_size"]

    print(f"Iniciando a construção da ABT rica com dados de 'previous_application'...")

    offset = 0
    is_first_chunk = True

    # Altere a query_base dentro do seu run_abt_generation no abt_transform.py:
    query_base = f"""
        SELECT 
            app.*,
            COALESCE(prev.prev_contract_count, 0) AS prev_contract_count,
            COALESCE(prev.prev_refused_count, 0) AS prev_refused_count,
            COALESCE(prev.prev_avg_amt_application, 0) AS prev_avg_amt_application
        FROM "{input_table}" app
        LEFT JOIN (
            SELECT 
                sk_id_curr,
                COUNT(sk_id_prev) AS prev_contract_count,
                SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END) AS prev_refused_count,
                AVG(amt_application) AS prev_avg_amt_application
            FROM previous_application_clean
            GROUP BY sk_id_curr
        ) prev ON app.sk_id_curr = prev.sk_id_curr
        LIMIT {chunk_size} OFFSET %s;
    """

    while True:
        # Passamos o offset de forma segura como parâmetro do cursor
        chunk_df = pd.read_sql(query_base, conn, params=(offset,))

        if chunk_df.empty:
            break

        print(f"Processando lote enriquecido (Offset: {offset}) para a ABT...")
        abt_chunk = build_features(chunk_df)

        if is_first_chunk:
            create_abt_table_schema(cursor, abt_chunk, output_table)
            conn.commit()
            is_first_chunk = False

        # Carga rápida padrão que você validou
        output = io.StringIO()
        abt_chunk.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        cursor.copy_expert(
            f'COPY "{output_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output
        )
        conn.commit()

        offset += chunk_size

    cursor.close()
    conn.close()
    print(f"--- ABT Enriquecida Construída com Sucesso! Tabela: '{output_table}' ---")


if __name__ == "__main__":
    run_abt_generation("postgres_data_db")