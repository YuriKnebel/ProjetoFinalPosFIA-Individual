# -*- coding: utf-8 -*-
"""
Módulo responsável pela construção da Analytical Base Table (ABT).

Este módulo utiliza a arquitetura ELT (Extract, Load, Transform), empurrando 
todo o processamento de agregações e JOINs para o motor do PostgreSQL.
Inclui a criação de índices intermediários e logs de volumetria.
"""
import os
import json
from utils import get_database_connection, log_row_count

# --- TASKS INTERMEDIÁRIAS (AGREGAÇÕES EM SQL NO BANCO) ---
def create_agg_previous_application(conn_id: str, output_prev_table: str):
    """Agrega o histórico de aplicações anteriores por cliente (sk_id_curr)."""
    tbl_dest = "tmp_prev_application_agg"
    conn = get_database_connection(conn_id)
    cur = conn.cursor()
    
    print(f"[AGREGAÇÃO] Processando '{output_prev_table}' -> '{tbl_dest}'...")
    log_row_count(cur, output_prev_table, "Base Histórica Bruta")
    
    cur.execute(f'DROP TABLE IF EXISTS "{tbl_dest}" CASCADE;')
    cur.execute(f"""
        CREATE TABLE "{tbl_dest}" AS
        SELECT
            sk_id_curr,
            SUM(CASE WHEN name_contract_status = 'Refused' THEN 1 ELSE 0 END)::float
                / NULLIF(COUNT(sk_id_prev), 0) AS prev_refused_rate
        FROM "{output_prev_table}"
        GROUP BY sk_id_curr;
    """)
    conn.commit()
    
    # Índice para a tabela temporária otimizar o JOIN final
    cur.execute(f'CREATE INDEX idx_tmp_prev_sk ON "{tbl_dest}" (sk_id_curr);')
    conn.commit()
    
    log_row_count(cur, tbl_dest, "Agregada Histórica (Clientes Únicos)")
    cur.close()
    conn.close()

def create_agg_bureau(conn_id: str, output_bureau_table: str):
    """Agrega os dados de birô de crédito por cliente (sk_id_curr)."""
    tbl_dest = "tmp_bureau_agg"
    conn = get_database_connection(conn_id)
    cur = conn.cursor()
    
    print(f"[AGREGAÇÃO] Processando '{output_bureau_table}' -> '{tbl_dest}'...")
    log_row_count(cur, output_bureau_table, "Base Bureau Bruta")
    
    cur.execute(f'DROP TABLE IF EXISTS "{tbl_dest}" CASCADE;')
    cur.execute(f"""
        CREATE TABLE "{tbl_dest}" AS
        SELECT
            sk_id_curr,
            COUNT(sk_id_bureau) AS bureau_credit_count,
            SUM(CASE WHEN credit_active = 'Active' THEN 1 ELSE 0 END)::float 
                / NULLIF(COUNT(sk_id_bureau), 0) AS bureau_credit_active_rate,
            AVG(amt_credit_sum) AS bureau_avg_credit_sum
        FROM "{output_bureau_table}"
        GROUP BY sk_id_curr;
    """)
    conn.commit()
    
    cur.execute(f'CREATE INDEX idx_tmp_bur_sk ON "{tbl_dest}" (sk_id_curr);')
    conn.commit()
    
    log_row_count(cur, tbl_dest, "Agregada Bureau (Clientes Únicos)")
    cur.close()
    conn.close()

def create_agg_installments(conn_id: str, output_installments_table: str):
    """Agrega o histórico de pagamentos de parcelas por cliente (sk_id_curr)."""
    tbl_dest = "tmp_installments_agg"
    conn = get_database_connection(conn_id)
    cur = conn.cursor()
    
    print(f"[AGREGAÇÃO] Processando '{output_installments_table}' -> '{tbl_dest}'...")
    log_row_count(cur, output_installments_table, "Base Parcelas Bruta")
    
    cur.execute(f'DROP TABLE IF EXISTS "{tbl_dest}" CASCADE;')
    cur.execute(f"""
        CREATE TABLE "{tbl_dest}" AS
        SELECT
            sk_id_curr,
            SUM(CASE WHEN days_entry_payment > days_instalment THEN 1 ELSE 0 END)::float 
                / NULLIF(COUNT(*), 0) AS inst_late_payment_rate
        FROM "{output_installments_table}"
        GROUP BY sk_id_curr;
    """)
    conn.commit()
    
    cur.execute(f'CREATE INDEX idx_tmp_inst_sk ON "{tbl_dest}" (sk_id_curr);')
    conn.commit()
    
    log_row_count(cur, tbl_dest, "Agregada Parcelas (Clientes Únicos)")
    cur.close()
    conn.close()

# --- GERAÇÃO FINAL DA ABT (SQL PURO, SEM CHUNKS) ---
def run_abt_generation(conn_id: str, config: dict):
    """
    Consolida todas as bases limpas e tabelas temporárias agregadas na ABT Final.
    
    Aplica a lógica de derivação de features finais (feature engineering) e imputação
    de zeros. Tudo via SQL nativo.

    Args:
        conn_id (str): Identificador da conexão.
        config (dict): Dicionário de configuração contendo parâmetros de banco.
    """
    db = config.get("database", {})
    clean_table = db.get("output_table", "application_clean")
    abt_table = db.get("abt_table", "application_abt")
    
    conn = get_database_connection(conn_id)
    cursor = conn.cursor()

    print(f"[ABT FINAL] Iniciando junção massiva em SQL (ELT) para gerar '{abt_table}'...")
    log_row_count(cursor, clean_table, "Entrada Application Clean")

    sql_elt = f"""
    DROP TABLE IF EXISTS "{abt_table}" CASCADE;
    
    CREATE TABLE "{abt_table}" AS
    SELECT 
        -- 1. Features Originais do Application Clean (Já tratadas no data_sanitization)
        a.*,
       
        -- 2. Engenharia de Features Derivadas (Relativas à Renda)
        (a.amt_credit / NULLIF(a.amt_income_total, 0)) AS fe_credit_income_percent,
        (a.amt_annuity / NULLIF(a.amt_income_total, 0)) AS fe_annuity_income_percent,
        
        -- 3. Features Agregadas do Previous Application
        COALESCE(p.prev_refused_rate, 0) AS prev_refused_rate,
        
        -- 4. Features Agregadas do Bureau
        CASE WHEN b.sk_id_curr IS NOT NULL THEN 1 ELSE 0 END AS has_bureau,
        COALESCE(b.bureau_credit_count, 0) AS bureau_credit_count,
        COALESCE(b.bureau_credit_active_rate, 0) AS bureau_credit_active_rate,
        COALESCE(b.bureau_avg_credit_sum, 0) AS bureau_avg_credit_sum,
        
        -- 5. Features Agregadas de Installments (Parcelas)
        CASE WHEN i.sk_id_curr IS NOT NULL THEN 1 ELSE 0 END AS has_installments_history,
        COALESCE(i.inst_late_payment_rate, 0) AS inst_late_payment_rate

    FROM "{clean_table}" a
    LEFT JOIN tmp_prev_application_agg p ON a.sk_id_curr = p.sk_id_curr
    LEFT JOIN tmp_bureau_agg b ON a.sk_id_curr = b.sk_id_curr
    LEFT JOIN tmp_installments_agg i ON a.sk_id_curr = i.sk_id_curr;
    """

    try:
        cursor.execute(sql_elt)
        conn.commit()
        
        print("[LIXEIRA] Limpando tabelas temporárias agregadas...")
        cursor.execute("DROP TABLE IF EXISTS tmp_prev_application_agg CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS tmp_bureau_agg CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS tmp_installments_agg CASCADE;")
        conn.commit()
        
        print(f"[SUCESSO] ABT finalizada com sucesso! Tabela materializada: '{abt_table}'")
        log_row_count(cursor, abt_table, "Saída ABT Final Gerada")
        
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Falha na geração ELT da ABT: {str(e)}")
    finally:
        cursor.close()
        conn.close()

