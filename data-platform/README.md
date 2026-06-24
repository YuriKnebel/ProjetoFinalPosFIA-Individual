###############################################
PARA INICIAR O CONTAINER:
###############################################

docker compose up -d --build

###############################################
PARA ACESSAR O CONTAINER DO AIRFLOW:
###############################################

docker compose exec airflow-webserver bash

##################################################
ACESSE O AIRFLOW 
##################################################

http://localhost:8080/

LOGIN: admin
SENHA: admin

