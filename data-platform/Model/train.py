import os
import joblib
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import lightgbm as lgb

def load_data_from_abt(conn_id: str) -> pd.DataFrame:
    """Busca a ABT final direto do banco de dados usando o Hook nativo do Postgres."""
    print("📥 Conectando ao banco via PostgresHook para extrair a ABT...")
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    
    # GARANTIA DE SUCESSO: Usando a conexão pura do psycopg2
    conn = pg_hook.get_conn()
    
    query = 'SELECT * FROM "application_abt";'
    df = pd.read_sql_query(query, conn)
    
    conn.close() # Fecha a conexão de forma limpa
    print(f"✅ ABT carregada com sucesso! Volumetria: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df

def preprocess_for_training(df: pd.DataFrame):
    """Separa as variáveis explicativas da meta (Target) e trata colunas categóricas."""
    print("⚙️ Preparando matrizes de treino e teste...")
    
    target_col = "target"
    cols_to_drop = [target_col, "sk_id_curr"]
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    y = df[target_col]
    
    # O LightGBM gerencia categorias nativamente se o tipo for 'category'
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category')
        
    return X, y

def train_model(conn_id, abt_table_name):
    """Busca a ABT especificada por parâmetro e treina o modelo."""
    from airflow.providers.postgres.hooks.postgres import PostgresHook
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import joblib
    import os
    
    print(f"📥 Extraindo dados da ABT: '{abt_table_name}' para o Modelo...")
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    
    # Query dinâmica apontando para o parâmetro correto
    query = f'SELECT * FROM "{abt_table_name}";'
    df_abt = pd.read_sql(query, conn)
    conn.close()

if __name__ == "__main__":
    train_model()