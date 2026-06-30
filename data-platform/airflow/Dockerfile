FROM apache/airflow:2.9.3

USER root
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# RETORNO PARA O USUÁRIO AIRFLOW ANTES DO PIP INSTALL E DA EXECUÇÃO
USER airflow

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# O modificador --chown garante que o usuário airflow seja o dono dos arquivos
RUN mkdir -p /opt/airflow/data/csv 
# && chown -R airflow:airflow /opt/airflow/data/csv