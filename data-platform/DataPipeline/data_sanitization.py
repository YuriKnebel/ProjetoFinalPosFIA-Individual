# -*- coding: utf-8 -*-
"""
data_sanitization_v2.py — Limpeza e padronização (Home Credit) no padrão da pipeline adaptada.

Unificação da lógica analítica v2 (estatísticas globais, tratamento de scores e outliers)
com o padrão arquitetural de conexões inteligentes e utilitários customizados.
"""
import os
import numpy as np
import pandas as pd
from utils import append_dataframe_to_postgres, get_database_connection, save_dataframe_to_postgres

def _reduz_cardinalidade(serie: pd.Series, min_freq: int) -> pd.Series:
    """Categorias com frequência < min_freq viram 'Other_low_freq'; nulos viram 'Unknown'."""
    print("Redução da Cardinalidade")
    freq = serie.value_counts()
    validas = freq[freq >= min_freq].index
    
    return (serie.where(serie.isin(validas), "Other_low_freq").fillna("Unknown").astype(str).str.strip())

# --------------------------------------------------------------------------
# Lógicas de Sanitização 
# --------------------------------------------------------------------------
def sanitize_application_train(df: pd.DataFrame, min_freq: int = 500, income_winsor_q: float = 0.99) -> pd.DataFrame:
    """Aplica a lógica exata de higienização v2 (Estatísticas Globais)."""
    print("Lendo DataFrame pandas")
    c = pd.DataFrame()

    print("Iniciando tratamento das Features da Application Train")
    c["sk_id_curr"] = (pd.to_numeric(df["sk_id_curr"], errors="coerce").astype("Int64"))
    c["target"] = pd.to_numeric(df["target"], errors="coerce").astype("Int64")

    # Scores externos: ext_source_2 + (1 e 3 imputados) + média combinada
    c["ext_source_2"] = (pd.to_numeric(df["ext_source_2"], errors="coerce").fillna(df["ext_source_2"].median()))
    _es = pd.concat(
        [
            pd.to_numeric(df["ext_source_1"], errors="coerce"),
            pd.to_numeric(df["ext_source_2"], errors="coerce"),
            pd.to_numeric(df["ext_source_3"], errors="coerce"),
        ],axis=1)
    c["ext_source_mean"] = _es.mean(axis=1)
    c["ext_source_mean"] = c["ext_source_mean"].fillna(c["ext_source_mean"].median())
    c["ext_source_1"] = _es.iloc[:, 0].fillna(_es.iloc[:, 0].median())
    c["ext_source_3"] = _es.iloc[:, 2].fillna(_es.iloc[:, 2].median())

    c["region_rating_client_w_city"] = (pd.to_numeric(df["region_rating_client_w_city"], errors="coerce").astype("Int64"))
    c["days_last_phone_change"] = (pd.to_numeric(df["days_last_phone_change"], errors="coerce").fillna(df["days_last_phone_change"].median()))
    c["days_id_publish"] = pd.to_numeric(df["days_id_publish"], errors="coerce")
    c["days_registration"] = pd.to_numeric(df["days_registration"], errors="coerce")

    for col in ["reg_city_not_work_city","reg_city_not_live_city","live_city_not_work_city",]:
        c[col] = (pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int))

    # own_car_age + flag has_car
    _own_car_age = pd.to_numeric(df["own_car_age"], errors="coerce")
    c["has_car"] = ((df["flag_own_car"].astype(str).str.strip() == "Y").astype(int))
    _median_car_age = _own_car_age[c["has_car"] == 1].median()
    c["own_car_age"] = (_own_car_age.where(c["has_car"] == 1, 0).fillna(_median_car_age))

    # Demais tratamentos numéricos
    c["def_60_cnt_social_circle"] = (pd.to_numeric(df["def_60_cnt_social_circle"], errors="coerce").fillna(0))
    c["amt_req_credit_bureau_year"] = (pd.to_numeric(df["amt_req_credit_bureau_year"], errors="coerce").fillna(0))
    c["cnt_children"] = (pd.to_numeric(df["cnt_children"], errors="coerce").fillna(0).astype(int))
    c["cnt_fam_members"] = (pd.to_numeric(df["cnt_fam_members"], errors="coerce").fillna(df["cnt_fam_members"].median()))

    # Renda: tratamento de outliers (Winsorização p99)
    income = pd.to_numeric(df["amt_income_total"], errors="coerce").replace(0, np.nan)
    income = income.fillna(income.median())
    p99 = income.quantile(income_winsor_q)
    c["amt_income_total"] = income.clip(upper=p99)

    c["amt_credit"] = pd.to_numeric(df["amt_credit"], errors="coerce")
    c["amt_annuity"] = (pd.to_numeric(df["amt_annuity"], errors="coerce").fillna(df["amt_annuity"].median()))

    # Categóricas e redução de cardinalidade
    c["occupation_type"] = (df["occupation_type"].fillna("Unknown").astype(str).str.strip())
    c["organization_type"] = _reduz_cardinalidade(df["organization_type"], min_freq)
    c["name_income_type"] = _reduz_cardinalidade(df["name_income_type"], min_freq)
    c["name_education_type"] = (df["name_education_type"].fillna("Unknown").astype(str).str.strip())
    c["code_gender"] = (df["code_gender"].replace("XNA", "Unknown").fillna("Unknown").astype(str).str.strip())

    print("Finalizando tratamento do DataFrame Pandas com as features sanitizadas")
    
    return c

def sanitize_bureau(df_bureau: pd.DataFrame) -> pd.DataFrame:
    """Tipagem e imputação para o histórico do Bureau (utilizada em transformações downstream)."""
    b = df_bureau.copy()
    b["credit_active"] = b["credit_active"].astype(str).str.strip()
    b["credit_type"] = b["credit_type"].astype(str).str.strip()

    cols_zero = [
        "amt_credit_sum",
        "amt_credit_sum_debt",
        "amt_credit_sum_overdue",
        "credit_day_overdue",
        "cnt_credit_prolong",
    ]
    for col in cols_zero:
        b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)

    for col in ["days_credit", "days_credit_update"]:
        b[col] = pd.to_numeric(b[col], errors="coerce")

    return b
# --------------------------------------------------------------------------
# Funções de Execução da Pipeline 
# --------------------------------------------------------------------------
def run_sanitization(conn_id: str, input_table: str, output_table: str, min_freq, winsor_q) -> None:
    """Executa a higienização da tabela principal carregando a base completa

    (necessário para o cálculo correto das métricas globais de mediana e quantis).
    """
    print(f"Carregando '{input_table}' completo para cálculo de estatísticas globais...")

    # Utilizando sua conexão inteligente camaleônica
    conn = get_database_connection(conn_id)

    print("Iniciando leitura dos dados")
    query = f'SELECT * FROM "{input_table}";'
    df = pd.read_sql_query(query, conn)

    # Processamento analítico
    print("--------- Iniciando sanitização de dados ------------------")
    clean_df = sanitize_application_train(df, min_freq=min_freq, income_winsor_q=winsor_q)
    nulos = int(clean_df.isna().sum().sum())
    print(f"Base limpa: {clean_df.shape[0]:,} linhas x {clean_df.shape[1]} colunas | nulos remanescentes={nulos}")

    print(f"Salvando dados na tabela destino: '{output_table}'...")
    save_dataframe_to_postgres(clean_df, output_table, conn_id)

    conn.close()
    print(f"--- Sanitização concluída! Tabela '{output_table}' atualizada. ---")

def run_prev_sanitization(conn_id: str, input_table: str, output_table: str, chunk_size: int) -> None:
    """Higieniza tabelas de histórico (como previous_application) processando via Chunks."""
    print(f"Sanitizando Histórico: {input_table} -> {output_table} em blocos de {chunk_size}...")
    conn = get_database_connection(conn_id)

    query = f'SELECT * FROM "{input_table}";'
    chunks = pd.read_sql_query(query, conn, chunksize=chunk_size)

    first_chunk = True
    for df_chunk in chunks:
        # Padronização básica de strings contida no seu script original
        if "name_contract_status" in df_chunk.columns:
            df_chunk["name_contract_status"] = (df_chunk["name_contract_status"].astype(str).str.strip())

        if first_chunk:
            save_dataframe_to_postgres(df_chunk, output_table, conn_id)
            first_chunk = False
        else:
            append_dataframe_to_postgres(df_chunk, output_table, conn_id)

    conn.close()
    print(f"Sanitização completa de {output_table} finalizada!")

def run_bureau_sanitization(conn_id: str, input_table: str, output_table: str, chunk_size: int) -> None:
    """Materializa e higieniza a tabela Bureau processando em chunks no seu padrão."""
    print(f"Materializando Bureau: {input_table} -> {output_table} em blocos de {chunk_size}...")
    conn = get_database_connection(conn_id)

    query = f'SELECT * FROM "{input_table}";'
    chunks = pd.read_sql_query(query, conn, chunksize=chunk_size)

    first_chunk = True
    for df_chunk in chunks:
        # Executa a limpeza exata do bureau v2 no pedaço atual
        df_chunk_clean = sanitize_bureau(df_chunk)

        if first_chunk:
            save_dataframe_to_postgres(df_chunk_clean, output_table, conn_id)
            first_chunk = False
        else:
            append_dataframe_to_postgres(df_chunk_clean, output_table, conn_id)

    conn.close()
    print(
        f"--- Bureau materializado com sucesso na tabela '{output_table}'! ---"
    )

if __name__ == "__main__":
    print("🚀 Executando pipeline de sanitização v2 direto no banco...")

    # 1. Altere com os dados reais do seu banco Postgres de teste/desenvolvimento
    # Formato: postgresql://usuario:senha@host:porta/nome_do_banco
    DB_URI = "postgresql://postgres:suasenha@localhost:5432/home_credit"

    # 2. Roda a sanitização da tabela principal (leitura inteira + estatísticas globais)
    run_sanitization(
        conn_id=DB_URI,
        input_table="application_train",  # Nome da sua tabela crua no banco
        output_table="application_clean",  # Tabela destino que será criada
    )

    # 3. Roda a materialização do Prep (em blocos de 50k linhas)
    run_prev_sanitization(
        conn_id=DB_URI,
        input_table="previous_application",  # Nome da sua tabela crua no banco
        output_table="previous_application_clean",  # Tabela destino que será criada
        chunk_size=50000,
    )

    # 4. Roda a materialização do Bureau (em blocos de 50k linhas)
    run_bureau_sanitization(
        conn_id=DB_URI,
        input_table="bureau",  # Nome da sua tabela crua no banco
        output_table="bureau_clean_v2",  # Tabela destino que será criada
        chunk_size=50000,
    )

    print("🏁 Execução concluída!")