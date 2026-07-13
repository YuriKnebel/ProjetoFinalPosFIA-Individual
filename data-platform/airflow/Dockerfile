FROM apache/airflow:2.9.3

USER root
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

RUN mkdir -p /opt/airflow/data/csv /opt/airflow/Model/artifacts \
    && chown -R airflow:0 /opt/airflow/data /opt/airflow/Model

# RETORNO PARA O USUÁRIO AIRFLOW ANTES DO PIP INSTALL E DA EXECUÇÃO
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
