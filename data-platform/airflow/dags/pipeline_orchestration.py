from datetime import datetime
import sys
import os
from airflow import DAG
from airflow.decorators import task

# Garante que o Airflow consiga encontrar e importar os scripts da pasta DataPipeline
sys.path.append("/opt/airflow/DataPipeline")
from data_sanitization import run_sanitization
from data_sanitization import run_prev_sanitization
from abt_transform import run_abt_generation

# Constantes centralizadas
CONN_ID = "postgres_data_db"

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 28),
    "retries": 0,
}

with DAG(
    "pipeline_orchestration",
    default_args=default_args,
    description="Orquestrador de Sanitização e ABT Nativo TaskFlow",
    schedule_interval=None,
    catchup=False,
    tags=["pipeline", "sanitization", "abt"],
) as dag:

    @task(task_id="data_sanitization")
    def task_sanitize(conn_id: str):
        # Chama a função mestre do script passando o ID da conexão nativa
        run_sanitization(conn_id)

    @task(task_id="clean_previous_application")
    def task_sanitize_prev(conn_id: str):
        run_prev_sanitization(conn_id)
    
    @task(task_id="abt_transform")
    def task_abt(conn_id: str):
        # Chama a função mestre do script da ABT passando o ID da conexão nativa
        run_abt_generation(conn_id)

    # Fluxo de execução nativo e limpo
    task_sanitize(CONN_ID) >> task_sanitize_prev(CONN_ID) >> task_abt(CONN_ID)