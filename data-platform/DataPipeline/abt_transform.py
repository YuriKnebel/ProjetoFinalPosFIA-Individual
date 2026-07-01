import io
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from utils import map_pandas_to_postgres_types

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

def run_abt_generation(conn_id, chunk_size, input_table, output_table, input_prev_table):
    """Gera a ABT cruzando as tabelas limpas e aplicando a engenharia de features por chunk."""
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()

    print(f" Construindo ABT '{output_table}' a partir de '{input_table}' e '{input_prev_table}'...")

    cursor.execute(f'DROP TABLE IF EXISTS "{output_table}" CASCADE;')
    print("Estrutura anterior destruída com sucesso.")

    query_base = f"""
        SELECT 
            app.*,
            COALESCE(prev.prev_contract_count, 0) AS prev_contract_count,
            COALESCE(prev.prev_refused_count, 0) AS prev_refused_count,
            COALESCE(prev.prev_avg_amt_approved, 0) AS prev_avg_amt_approved,
            COALESCE(prev.prev_avg_amt_refused, 0) AS prev_avg_amt_refused
        FROM "{input_table}" app
        LEFT JOIN (
            SELECT 
                sk_id_curr,
                COUNT(sk_id_prev) AS prev_contract_count,
                SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END) AS prev_refused_count,
                AVG(CASE WHEN name_contract_status = 'Approved' THEN amt_application END) AS prev_avg_amt_approved,
                AVG(CASE WHEN name_contract_status = 'Refused' THEN amt_application END) AS prev_avg_amt_refused
            FROM "{input_prev_table}"
            GROUP BY sk_id_curr
        ) prev ON app.sk_id_curr = prev.sk_id_curr
        LIMIT {chunk_size} OFFSET %s;
    """

    offset = 0
    first_chunk = True

    while True:
        # 1. Leitura do lote bruto vindo do banco
        chunk_df = pd.read_sql_query(query_base, conn, params=[offset])
        print("--------------- Leitura dos dados -----------------------")

        if chunk_df.empty:
            print(" Não há mais dados para ler.")
            break

        # 2. A MÁGICA ACONTECE AQUI: Enriquecimento do lote antes de salvar
        print(" Aplicando Engenharia de Features no Bloco...")
        chunk_df = build_features(chunk_df)

        # 3. Mapeamento de tipos dinâmicos (considerando as novas colunas criadas!)
        if first_chunk:
            colunas_sql = map_pandas_to_postgres_types(chunk_df)
            cursor.execute(f'CREATE TABLE "{output_table}" ({", ".join(colunas_sql)});')
            first_chunk = False

        # 4. Inserção Ultra Rápida via COPY
        output = io.StringIO()
        chunk_df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)
        cursor.copy_expert(f'COPY "{output_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output,)
        conn.commit()

        print(f"Processado Bloco de {len(chunk_df)} linhas (Offset: {offset})")
        offset += chunk_size

    cursor.close()
    conn.close()
    print("-------- Geração da ABT e Engenharia de Features concluídas! ---------")