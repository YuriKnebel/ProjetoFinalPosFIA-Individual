from datetime import datetime
import sys
import os
import json

# Mapeamento dos caminhos do projeto
sys.path.append("/opt/airflow/DataPipeline")
sys.path.append("/opt/airflow/modelos")

from ingestion import run_csv_ingestion
from data_sanitization import run_sanitization, run_prev_sanitization, run_bureau_sanitization
from abt_transform import create_agg_previous_application, create_agg_bureau, create_agg_installments, run_abt_generation
from train import train_model

from airflow import DAG
from airflow.decorators import task

# Parâmetros de Infraestrutura imutáveis
CONN_ID = "postgres_data_db"
PASTA_DATA = "/opt/airflow/data/csv"
CONFIG_PATH = "/opt/airflow/DataPipeline/config_pipeline.json"

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Extração das chaves do JSON para distribuição nas tasks
tabelas_para_ingerir = config.get("ingestion_table", {}).get("using_csv", [])
db_config = config.get("database", {})
clean_params = config.get("cleaning_parameters", {})
bureau_features_config = config.get("BUREAU_FEATURE_COLS", [])
sanitization_params = config.get("sanitization", {})

with DAG(
    dag_id="pipeline_orchestration",
    start_date=datetime(2026, 6, 24),
    schedule=None,
    catchup=False,
) as dag:

    # --- TASK DE INGESTÃO ---
    @task(task_id="ingest_csv_source", pool="pool_ingestao")
    def task_ingest(config_tabela: dict, conn_id: str, pasta_origem: str, config_file: str):
        
        nome_tabela = config_tabela["table_name"]
        tamanho_chunk = config_tabela["chunk_size"]
        
        print(f"Iniciando carga da tabela '{nome_tabela}' com chunksize de {tamanho_chunk}...")
        
        run_csv_ingestion(
            pasta_origem=pasta_origem, 
            table_name=nome_tabela, 
            conn_id=conn_id, 
            config_file=config_file, 
            chunk_size=tamanho_chunk
        )

    # --- TASKS DE HIGIENIZAÇÃO ---
    @task(task_id="sanitize_application_train", pool="pool_sanitization")
    def task_sanitize_app(conn_id: str, input_t: str, output_t: str, min_freq: int, winsor_q: float):
        run_sanitization(conn_id, input_t, output_t, min_freq, winsor_q)

    @task(task_id="sanitize_previous_application", pool="pool_sanitization")
    def task_sanitize_prev(conn_id: str, input_t: str, output_t: str, chunk: int):
        run_prev_sanitization(conn_id, input_t, output_t, chunk)

    @task(task_id="sanitize_bureau", pool="pool_sanitization")
    def task_sanitize_bureau(conn_id: str, input_t: str, output_t: str, chunk: int):
        run_bureau_sanitization(conn_id, input_t, output_t, chunk)

    # --- TASKS INTERMEDIÁRIAS PARA AGREGACAO (PROCESSADAS VIA SQL NO BANCO) ---
    @task(task_id="agg_intermediate_prev", pool="pool_aggregation")
    def task_agg_prev(conn_id: str, output_prev_table: str):
        create_agg_previous_application(conn_id, output_prev_table)

    @task(task_id="agg_intermediate_bureau", pool="pool_aggregation")
    def task_agg_bureau(conn_id: str, output_bureau_table: str):
        create_agg_bureau(conn_id, output_bureau_table)

    @task(task_id="agg_intermediate_installments", pool="pool_aggregation")
    def task_agg_inst(conn_id: str, output_installments_table: str):
        create_agg_installments(conn_id, output_installments_table)

    # --- TASK DA CONSTRUÇÃO DA ABT EM LOTES ---
    @task(task_id="generate_analytical_base_table")
    def task_abt(conn_id: str, bureau_feature_cols: list, clean_table: str, abt_table: str, chunk_size: int, key_col: str):
        run_abt_generation(
            conn_id=conn_id,
            bureau_feature_cols=bureau_feature_cols,
            clean_table=clean_table,
            abt_table=abt_table,
            chunk_size=chunk_size,
            key_col=key_col
        )

    # --- TASK DE TREINAMENTO ---
    @task(task_id="train_machine_learning_model")
    def task_train(conn_id: str, abt_table: str):
        train_model(conn_id, abt_table)

    # --- INSTANCIANDO AS TAREFAS ---
    carga_inicial = task_ingest.partial(
        conn_id=CONN_ID, 
        pasta_origem=PASTA_DATA, 
        config_file=CONFIG_PATH
    ).expand(config_tabela=tabelas_para_ingerir)

    limpeza_app = task_sanitize_app(
        conn_id=CONN_ID,
        input_t=db_config.get("input_table"),
        output_t=db_config.get("output_table"),
        min_freq=sanitization_params.get("cardinalidade_min_freq", 500),
        winsor_q=sanitization_params.get("income_winsor_q", 0.99)
    )

    limpeza_prev = task_sanitize_prev(
        conn_id=CONN_ID,
        input_t=db_config.get("input_prev_table"),
        output_t=db_config.get("output_prev_table"),
        chunk=clean_params.get("chunk_size", 50000)
    )

    limpeza_bureau = task_sanitize_bureau(
        conn_id=CONN_ID,
        input_t=db_config.get("input_bureau_table"),
        output_t=db_config.get("output_bureau_table"),
        chunk=clean_params.get("chunk_size", 50000)
    )

    t_prev = task_agg_prev(CONN_ID, db_config.get("output_prev_table"))
    t_bureau = task_agg_bureau(CONN_ID, db_config.get("output_bureau_table"))
    t_inst = task_agg_inst(CONN_ID, db_config.get("output_installments_table", "installments_clean"))

    t_abt_final = task_abt(
        conn_id=CONN_ID,
        bureau_feature_cols=bureau_features_config,
        clean_table=db_config.get("output_table"),
        abt_table=db_config.get("abt_table"),
        chunk_size=clean_params.get("chunk_size", 50000),
        key_col=config.get("indexes", {}).get(db_config.get("input_table"), "sk_id_curr")
    )

    treino_modelo = task_train(CONN_ID, db_config.get("abt_table"))

    # --- DEFINIÇÃO DO FLUXO (DEPENDÊNCIAS) ---
    carga_inicial >> [limpeza_app, limpeza_prev, limpeza_bureau]
    
    limpeza_prev >> t_prev
    limpeza_bureau >> [t_bureau, t_inst]
    
    [t_prev, t_bureau, t_inst, limpeza_app] >> t_abt_final >> treino_modelo