from datetime import datetime
import sys
import os
import json

# Mapeamento dos caminhos do projeto
sys.path.append("/opt/airflow/DataPipeline")
sys.path.append("/opt/airflow/modelos")

from ingestion import run_csv_ingestion
from data_sanitization import run_sanitization, run_prev_sanitization, run_bureau_sanitization
from abt_transform import run_abt_generation
from train import train_model

from airflow import DAG
from airflow.decorators import task

# Parâmetros de Infraestrutura imutáveis
CONN_ID = "postgres_data_db"
PASTA_DATA = "/opt/airflow/data/csv"
CONFIG_PATH = "/opt/airflow/DataPipeline/config_pipeline.json"


with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Extração organizada das chaves do JSON para distribuição
tabelas_para_ingerir = config.get("ingestion_table", {}).get("using_csv", [])
db_config = config.get("database", {})
clean_params = config.get("cleaning_parameters", {})
bureau_features_config = config.get("BUREAU_FEATURE_COLS", {})
sanitization_params = config.get("sanitization")

with DAG(
    dag_id="pipeline_orchestration",
    start_date=datetime(2026, 6, 24),
    schedule=None,
    catchup=False,
) as dag:

    @task(task_id="ingest_csv_source")
    def task_ingest(table_name: str, conn_id: str, pasta_origem: str, config_file: str):
        run_csv_ingestion(table_name, conn_id, pasta_origem, config_file)

    @task(task_id="sanitize_application_train")
    def task_sanitize_app(conn_id: str, input_t: str, output_t: str, min_freq, winsor_q):
        run_sanitization(conn_id, input_t, output_t, min_freq, winsor_q)

    @task(task_id="sanitize_previous_application")
    def task_sanitize_prev(conn_id: str, input_t: str, output_t: str, chunk: int):
        run_prev_sanitization(conn_id, input_t, output_t, chunk)

    @task(task_id="run_bureau_sanitization")
    def task_sanitize_bureau(conn_id: str, input_t: str, output_t: str, chunk: int):
        run_bureau_sanitization(conn_id, input_t, output_t, chunk)

    @task(task_id="generate_analytical_base_table")
    def task_abt(conn_id: str, chunk: int, output_t: str, input_t: str, input_prev_t: str):
        run_abt_generation(conn_id, chunk, input_table=input_t, output_table=output_t, input_prev_table=input_prev_t)

    @task(task_id="train_machine_learning_model")
    def task_train(conn_id: str, abt_table: str):
        train_model(conn_id, abt_table_name=abt_table)


    # Carga Dinâmica via Mapeamento Nativo
    carga_inicial = task_ingest.partial(
        conn_id=CONN_ID, 
        pasta_origem=PASTA_DATA, 
        config_file=CONFIG_PATH
    ).expand(table_name=tabelas_para_ingerir)

    # Instanciando as tarefas com injeção direta de dependências do JSON
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
        chunk=clean_params.get("chunk_size")
    )
    limpeza_bureau = task_sanitize_bureau(
        conn_id=CONN_ID,
        input_t=db_config.get("input_bureau_table"),
        output_t=db_config.get("output_bureau_table"),
        chunk=clean_params.get("chunk_size")
    )
    
    construcao_abt = task_abt(
        conn_id=CONN_ID,
        chunk=clean_params.get("chunk_size"),
        output_t=db_config.get("abt_table"),
        input_t=db_config.get("output_table"),       
        input_prev_t=db_config.get("output_prev_table") 
    )
    
    treino_lgbm = task_train(
        conn_id=CONN_ID,
        abt_table=db_config.get("abt_table")
    )

    # --- ORQUESTRAÇÃO DAS TASKS ---
    carga_inicial >> [limpeza_app, limpeza_prev, limpeza_bureau]
    [limpeza_app, limpeza_prev], limpeza_bureau >> construcao_abt >> treino_lgbm