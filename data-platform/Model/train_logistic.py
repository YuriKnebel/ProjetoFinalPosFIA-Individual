import os
import joblib
import pandas as pd
import numpy as np
from airflow.providers.postgres.hooks.postgres import PostgresHook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

def load_data_from_abt(conn_id: str) -> pd.DataFrame:
    """Busca a ABT final direto do banco de dados de forma nativa."""
    pg_hook = PostgresHook(postgres_conn_id=conn_id)
    conn = pg_hook.get_conn()
    
    query = 'SELECT * FROM "application_abt";'
    df = pd.read_sql(query, conn)
    
    conn.close()
    return df

def preprocess_for_logistic(df: pd.DataFrame):
    """Pré-processamento estrito necessário para modelos lineares como Regressão Logística."""
    print("⚙️ Aplicando transformações lineares na ABT...")
    
    target_col = "target"
    X = df.drop(columns=[col for col in [target_col, "sk_id_curr"] if col in df.columns])
    y = df[target_col]
    
    # 1. Regressão Logística NÃO aceita texto. Transformamos categorias em colunas 0 e 1 (Dummies)
    X = pd.get_dummies(X, drop_first=True)
    
    # 2. Regressão Logística NÃO aceita Nulos (NaN). Vamos preencher com a mediana de cada coluna
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
            
    return X, y

def train_logistic_model(conn_id: str = "postgres_data_db"):
    """Treina o modelo alternativo de Regressão Logística para Baseline."""
    df_abt = load_data_from_abt(conn_id)
    X, y = preprocess_for_logistic(df_abt)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("🚀 Treinando Regressão Logística com peso de classe balanceado...")
    # 'class_weight='balanced'' resolve o desbalanceamento de 8% de forma equivalente ao LightGBM
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Avaliação
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_label = model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n🎯 Baseline (Regressão Logística) - AUC-ROC em Teste: {auc_score:.4f}")
    
    print("\n📋 Relatório de Classificação Detalhado:")
    print(classification_report(y_test, y_pred_label))
    
    # Salvando o modelo na pasta correta
    model_dir = "/opt/airflow/Model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "logistic_credit_model.pkl"))
    print("💾 Modelo alternativo salvo com sucesso!")

if __name__ == "__main__":
    train_logistic_model()