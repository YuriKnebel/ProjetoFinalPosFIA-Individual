import io
import json
import os
import numpy as np
import pandas as pd
from utils import get_database_connection, map_pandas_to_postgres_types

# --------------------------------------------------------------------------
# Lógicas de Engenharia de Features e Agregações 
# --------------------------------------------------------------------------
def build_features_application(df_app: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas do application_train (idade, tempo de emprego, razões)."""
    
    print(f"------ Inicializando build Features application_train ------------------------------------")
    f = pd.DataFrame()
    f["sk_id_curr"] = (pd.to_numeric(df_app["sk_id_curr"], errors="coerce").astype("Int64"))
    f["age"] = (np.abs(pd.to_numeric(df_app["days_birth"], errors="coerce"))/ 365.25)

    _days_emp = pd.to_numeric(df_app["days_employed"], errors="coerce")
    f["years_employed"] = (np.abs(_days_emp.replace(365243, np.nan)).fillna(0) / 365.25)
    f["days_employed_anom"] = (_days_emp == 365243).astype(int)

    credit = pd.to_numeric(df_app["amt_credit"], errors="coerce")
    income = pd.to_numeric(df_app["amt_income_total"], errors="coerce").replace(0, np.nan)
    annuity = pd.to_numeric(df_app["amt_annuity"], errors="coerce")
    f["fe_credit_income_percent"] = credit / income
    f["fe_annuity_income_percent"] = annuity / income
    ratios = ["fe_credit_income_percent", "fe_annuity_income_percent"]
    f[ratios] = f[ratios].fillna(f[ratios].median())

    print(f"------ Finalizando build Features application_train ------------------------------------")
    
    return f

def aggregate_previous_application(df_prev: pd.DataFrame) -> pd.DataFrame:
    """Agrega previous_application para 1 linha por cliente."""

    print("------ Inicializando agreggation Features previous_application ------------------------------------")
    
    df_prev = df_prev.copy()
    df_prev["name_contract_status"] = (df_prev["name_contract_status"].astype(str).str.strip().str.title())
    print("Agregando dados - 1 linha por cliente")
    agg = (
        df_prev.groupby("sk_id_curr").agg(
            prev_contract_count=("sk_id_prev", "count"),
            prev_refused_count=("name_contract_status", lambda x: (x == "Refused").sum()),
            prev_approved_count=("name_contract_status", lambda x: (x == "Approved").sum()),
            prev_avg_amt_application=("amt_application", "mean"),
            prev_avg_amt_credit=("amt_credit", "mean"),
            prev_avg_annuity=("amt_annuity", "mean"),
            prev_max_annuity=("amt_annuity", "max"),
            prev_avg_down_payment=("amt_down_payment", "mean"),
            prev_avg_cnt_payment=("cnt_payment", "mean"),
            prev_last_days_decision=("days_decision", "max"),
        )
        .reset_index()
    )
    agg["prev_refused_rate"] = agg["prev_refused_count"] / agg["prev_contract_count"]

    print("------ Finalizando agreggation Features previous_application ------------------------------------")                            

    return agg

def build_prev_features(df_prev_agg: pd.DataFrame, df_app_ids: pd.DataFrame) -> pd.DataFrame:
    """Seleciona prev_refused_rate + flag has_prev_app, imputando quem não tem histórico."""
    
    print("------ Inicializando build Features previous_application ------------------------------------")
    media = df_prev_agg["prev_refused_rate"].mean()
    feats = df_app_ids[["sk_id_curr"]].merge(
        df_prev_agg[["sk_id_curr", "prev_refused_rate"]],
        on="sk_id_curr",
        how="left",
    )
    feats["has_prev_app"] = feats["prev_refused_rate"].notna().astype(int)
    feats["prev_refused_rate"] = feats["prev_refused_rate"].fillna(media)
    
    print("------ Finalizando build Features previous_application ------------------------------------")    
    
    return feats

def aggregate_bureau(df_bureau_clean: pd.DataFrame) -> pd.DataFrame:
    """Agrega bureau para 1 linha por cliente."""
    
    print("------ Inicializando agreggation Features bureau ------------------------------------")
    agg = (df_bureau_clean.groupby("sk_id_curr").agg(
            bureau_credit_count=("sk_id_bureau", "count"),
            bureau_active_count=("credit_active", lambda x: (x == "Active").sum()),
            bureau_closed_count=("credit_active", lambda x: (x == "Closed").sum()),
            bureau_bad_debt_count=("credit_active", lambda x: (x == "Bad debt").sum()),
            bureau_sold_count=("credit_active", lambda x: (x == "Sold").sum()),
            bureau_total_credit=("amt_credit_sum", "sum"),
            bureau_avg_credit=("amt_credit_sum", "mean"),
            bureau_total_debt=("amt_credit_sum_debt", "sum"),
            bureau_avg_debt=("amt_credit_sum_debt", "mean"),
            bureau_total_overdue=("amt_credit_sum_overdue", "sum"),
            bureau_max_overdue=("amt_credit_sum_overdue", "max"),
            bureau_max_days_overdue=("credit_day_overdue", "max"),
            bureau_overdue_count=("credit_day_overdue", lambda x: (x > 0).sum()),
            bureau_total_prolong=("cnt_credit_prolong", "sum"),
            bureau_avg_days_credit=("days_credit", "mean"),
            bureau_last_days_credit=("days_credit", "max"),
            bureau_avg_days_credit_update=("days_credit_update", "mean"),
            bureau_last_days_credit_update=("days_credit_update", "max"),
        )
        .reset_index()
    )
    agg["bureau_active_rate"] = (agg["bureau_active_count"] / agg["bureau_credit_count"])
    agg["bureau_closed_rate"] = (agg["bureau_closed_count"] / agg["bureau_credit_count"])
    agg["bureau_debt_credit_ratio"] = agg["bureau_total_debt"] / agg["bureau_total_credit"].replace(0, np.nan)
    agg["bureau_overdue_credit_ratio"] = agg["bureau_total_overdue"] / agg["bureau_total_credit"].replace(0, np.nan)
    agg[["bureau_debt_credit_ratio", "bureau_overdue_credit_ratio"]] = (agg[["bureau_debt_credit_ratio", "bureau_overdue_credit_ratio"]].fillna(0))
    agg["bureau_debt_credit_ratio"] = agg["bureau_debt_credit_ratio"].clip(lower=-1, upper=1)
    
    print("------ Finalizando agreggation Features bureau ------------------------------------") 
 
    return agg

def build_abt(df_app_clean: pd.DataFrame, feats_app: pd.DataFrame, feats_prev: pd.DataFrame, feats_bureau: pd.DataFrame, bureau_feature_cols: list) -> pd.DataFrame:
    """Junta as fontes (1 linha por cliente) + tratamentos da ABT v2."""
    
    # Padronização explícita dos tipos da chave de cruzamento para evitar o erro de object e Int64
    df_app_clean = df_app_clean.copy()
    df_app_clean["sk_id_curr"] = pd.to_numeric(df_app_clean["sk_id_curr"], errors="coerce").astype("Int64")
    
    feats_app = feats_app.copy()
    feats_app["sk_id_curr"] = pd.to_numeric(feats_app["sk_id_curr"], errors="coerce").astype("Int64")
    
    feats_prev = feats_prev.copy()
    feats_prev["sk_id_curr"] = pd.to_numeric(feats_prev["sk_id_curr"], errors="coerce").astype("Int64")
    
    feats_bureau = feats_bureau.copy()
    feats_bureau["sk_id_curr"] = pd.to_numeric(feats_bureau["sk_id_curr"], errors="coerce").astype("Int64")

    abt = (
        df_app_clean.merge(feats_app, on="sk_id_curr", how="left")
        .merge(feats_prev, on="sk_id_curr", how="left")
        .merge(feats_bureau, on="sk_id_curr", how="left")
    )
    abt["has_bureau"] = abt["bureau_active_count"].notna().astype(int)
    abt[bureau_feature_cols] = abt[bureau_feature_cols].fillna(0)
    return abt

def check_abt_integrity(abt: pd.DataFrame, df_app_clean: pd.DataFrame) -> None:
    """Gate de qualidade da ABT (Padrão de Validação do Modelo)."""
    assert bool(abt["sk_id_curr"].is_unique), "sk_id_curr duplicado na ABT"
    assert (abt.shape[0] == df_app_clean["sk_id_curr"].nunique()), "n de linhas != n de clientes"
    nulos = int(abt.isna().sum().sum())
    dist = ((abt["target"].value_counts(normalize=True) * 100).round(2).to_dict())
    print(f"Integridade ABT: 1 linha/cliente OK | linhas={abt.shape[0]:,} | colunas={abt.shape[1]} | nulos={nulos} | target%={dist}")
    assert nulos == 0, f"Há {nulos} valores nulos residuais na ABT!"

# --------------------------------------------------------------------------
# Função Mestre de Geração da ABT (Seu Padrão de Execução Isolado)
# --------------------------------------------------------------------------
def run_abt_generation(conn_id: str, bureau_feature_cols: list, clean_table: str, input_table: str, prev_table: str, bureau_table: str, abt_table: str) -> None:
    """Monta a ABT rica cruzando as tabelas e aplicando a engenharia v2."""

    # Utilizando sua conexão inteligente e isolada
    conn = get_database_connection(conn_id)
    cursor = conn.cursor()

    print(f"Construindo ABT v2 '{abt_table}' a partir de fontes integradas...")

    # 1. Leitura Completa das fontes para garantir consistência dos Joins/Agregações
    print("Carregando dados do banco para a memória...")
    df_clean = pd.read_sql_query(f'SELECT * FROM "{clean_table}"', conn)

    query_app = f"""SELECT sk_id_curr, days_birth, days_employed, amt_credit, amt_income_total, amt_annuity, cnt_fam_members 
                      FROM "{input_table}"
                """
    df_app = pd.read_sql_query(query_app, conn)

    query_prev = f"""SELECT sk_id_curr, sk_id_prev, name_contract_status, amt_application, amt_credit,
                    amt_annuity, amt_down_payment, cnt_payment, days_decision FROM "{prev_table}" """
    df_prev = pd.read_sql_query(query_prev, conn)

    df_bureau = pd.read_sql_query(f'SELECT * FROM "{bureau_table}"', conn)

    # Forçando cast dos dataframes de entrada para garantir homogeneidade na agregação interna
    df_app["sk_id_curr"] = pd.to_numeric(df_app["sk_id_curr"], errors="coerce").astype("Int64")
    df_prev["sk_id_curr"] = pd.to_numeric(df_prev["sk_id_curr"], errors="coerce").astype("Int64")
    df_bureau["sk_id_curr"] = pd.to_numeric(df_bureau["sk_id_curr"], errors="coerce").astype("Int64")

    # 2. Processamento Analítico e Engenharia de Features v2
    print("[v2] Executando pipeline automática de agregação...")
    feats_app = build_features_application(df_app)
    feats_prev = build_prev_features(
        aggregate_previous_application(df_prev), df_app[["sk_id_curr"]]
    )

    df_bureau_clean = pd.read_sql_query(f'SELECT * FROM "{bureau_table}"', conn)
    df_bureau_clean["sk_id_curr"] = pd.to_numeric(df_bureau_clean["sk_id_curr"], errors="coerce").astype("Int64")

    feats_bureau = aggregate_bureau(df_bureau_clean)
    # Seleção de colunas baseada na lista parametrizável enviada pela DAG
    feats_bureau = feats_bureau[["sk_id_curr"] + bureau_feature_cols].copy()

    # 3. Montagem e Validação da ABT Final
    print("[v2] Consolidando ABT e executando validações de integridade...")
    abt = build_abt(df_clean, feats_app, feats_prev, feats_bureau, bureau_feature_cols)
    check_abt_integrity(abt, df_clean)

    # 4. Destruição da estrutura antiga e mapeamento dinâmico de tipos 
    print(f"[v2] Preparando estrutura na tabela destino '{abt_table}'...")
    cursor.execute(f'DROP TABLE IF EXISTS "{abt_table}" CASCADE;')

    colunas_sql = map_pandas_to_postgres_types(abt)
    cursor.execute(f'CREATE TABLE "{abt_table}" ({", ".join(colunas_sql)});')
    conn.commit()

    # 5. Inserção Ultra Rápiva via COPY EXPERT
    print(f"[v2] Injetando {len(abt):,} linhas via COPY...")
    output = io.StringIO()
    abt.to_csv(output, sep="\t", header=False, index=False)
    output.seek(0)

    cursor.copy_expert(f'COPY "{abt_table}" FROM STDIN WITH CSV DELIMITER \'\t\' NULL \'\'', output)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"-------- Geração da ABT Concluída com Sucesso! Tabela '{abt_table}' criada. --------")