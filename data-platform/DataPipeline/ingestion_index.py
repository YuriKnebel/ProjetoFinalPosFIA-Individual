from utils import get_database_connection

# ---------------------------------------------------------------------------
# Task: Criação de Índices Otimizados (Rodar ANTES das limpezas)
# ---------------------------------------------------------------------------
def run_create_indexes(conn_id: str):
    """
    Cria índices nas tabelas raw para otimizar as operações de limpeza e futuros JOINs.
    """
    conn = get_database_connection(conn_id)
    cursor = conn.cursor()
    
    print("--- Criando índices para otimizar leitura nas tabelas de origem. ---")

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_app_sk_id_curr ON application_train (sk_id_curr);",
        "CREATE INDEX IF NOT EXISTS idx_app_org_type ON application_train (organization_type);",
        "CREATE INDEX IF NOT EXISTS idx_app_inc_type ON application_train (name_income_type);",
        "CREATE INDEX IF NOT EXISTS idx_app_flag_car ON application_train (flag_own_car);",
        "CREATE INDEX IF NOT EXISTS idx_prev_sk_id_prev ON previous_application (sk_id_prev);",
        "CREATE INDEX IF NOT EXISTS idx_prev_sk_id_curr ON previous_application (sk_id_curr);",
        "CREATE INDEX IF NOT EXISTS idx_bur_sk_id_bureau ON bureau (sk_id_bureau);",
        "CREATE INDEX IF NOT EXISTS idx_bur_sk_id_curr ON bureau (sk_id_curr);",
        "CREATE INDEX IF NOT EXISTS idx_inst_sk_id_curr ON installments_payments (sk_id_curr);",
        "CREATE INDEX IF NOT EXISTS idx_inst_sk_id_prev ON installments_payments (sk_id_prev);"
    ]

    for sql in indexes:
        cursor.execute(sql)
        conn.commit()

    cursor.close()
    conn.close()
    print("--- Índices criados com sucesso! Banco pronto para processamento ELT. ---")