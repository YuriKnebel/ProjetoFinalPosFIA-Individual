# Predição de risco de crédito — Projeto Final FIA Labdata

Projeto final da pós-graduação em Engenharia de Dados da FIA Labdata. A solução
constrói uma base analítica de clientes, treina um modelo de classificação para
risco de inadimplência e disponibiliza o score por script, API e interface web.

Os dados são baseados na competição
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

## Objetivo de negócio

Apoiar a análise de crédito com um sinal consistente de risco, reduzindo perdas
por inadimplência sem recusar indiscriminadamente clientes adimplentes. O modelo
ordena os clientes por risco; a decisão final continua separada em uma política
de crédito, sujeita a regras cadastrais, fraude, capacidade financeira e revisão
humana.

## Metodologia

1. Ingestão dos CSVs no PostgreSQL por DAG do Apache Airflow.
2. Limpeza e padronização em lotes, preservando os dados brutos.
3. Construção da ABT, com uma linha por cliente e histórico consolidado.
4. Análise exploratória dos dados limpos.
5. Separação estratificada entre treino e teste.
6. Pipeline de imputação, padronização, one-hot encoding e regressão logística.
7. Otimização por `GridSearchCV`, usando ROC AUC e validação cruzada.
8. Avaliação em teste isolado e análise de interpretabilidade por coeficientes.
9. Persistência do pré-processamento e do modelo em um único artefato Pickle.

## Estrutura exigida pelo item C

```text
ProjetoFinalPosFIA/
├── README.md
└── data-platform/
    ├── Dados/
    │   ├── raw_data.csv         # materializado localmente; não versionado
    │   ├── clean_data.csv       # materializado localmente; não versionado
    │   ├── abt.csv              # materializado localmente; não versionado
    │   └── materialize.py
    ├── DataPipeline/
    │   ├── data_sanitization.py
    │   ├── abt_transform.py
    │   ├── exp_analysis.ipynb
    │   └── config_pipeline.json # variáveis, parâmetros e metadados
    ├── Model/
    │   ├── train.py
    │   ├── evaluation.ipynb
    │   ├── predict.py
    │   ├── config_model.json    # variáveis, parâmetros e metadados
    │   └── artifacts/
    ├── MLOps/                   # app, testes, Dockerfiles e orquestração
    ├── docker-compose.yml
    └── requirements.txt
```

Os CSVs completos têm dezenas ou centenas de megabytes e são ignorados pelo
Git. [Dados/README.md](data-platform/Dados/README.md) documenta sua origem e
reprodução.

## Configuração do ambiente

Na raiz do repositório:

```bash
cd data-platform
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Coloque os arquivos originais do Home Credit em `data-platform/data/csv`.

## Pipeline de dados

```bash
cd data-platform
docker compose up -d --build postgres airflow-init airflow-webserver airflow-scheduler jupyter
```

- Airflow: `http://localhost:8080` (`admin` / `admin`)
- Jupyter: `http://localhost:8888` (token configurado no `.env`)

Execute primeiro a DAG `loadfile_csv_to_postgres` e depois
`pipeline_orchestration`. A análise exploratória está em
`DataPipeline/exp_analysis.ipynb`.

## Como treinar o modelo

Na pasta `data-platform`:

```bash
.venv/bin/python Model/train.py
```

O script lê `Model/config_model.json` e grava o artefato em
`Model/artifacts/logistic_regression_abt.pkl`. Para um teste rápido sem
substituir o modelo oficial:

```bash
.venv/bin/python Model/train.py \
  --sample-size 5000 \
  --n-jobs 1 \
  --output /tmp/logistic_regression_smoke.pkl
```

## Comparação de modelos

Para justificar a escolha do algoritmo, execute a comparação entre Regressão
Logística, Árvore de Decisão e Random Forest usando a mesma ABT, o mesmo split e
as métricas `roc_auc`, `accuracy` e `precision`:

```bash
.venv/bin/python Model/compare_models.py
```

O resultado é salvo em `Model/artifacts/model_comparison.csv`. Para um teste
rápido:

```bash
.venv/bin/python Model/compare_models.py --sample-size 5000 --n-jobs 1
```

Na justificativa técnica, priorize `roc_auc`, pois a base é desbalanceada e o
objetivo principal é ordenar clientes por risco.

## Avaliação e interpretabilidade

Abra e execute `Model/evaluation.ipynb`. O notebook reproduz o conjunto de
teste, calcula ROC AUC, Average Precision, relatório de classificação, matriz de
confusão e curvas ROC/Precisão–Recall, além de ordenar os coeficientes de maior
influência.

## Predição local

```bash
.venv/bin/python Model/predict.py --input Dados/abt.csv --row 0
```

Também é possível fornecer um JSON contendo todas as features esperadas.

## Execução do serviço de predição

### Docker Compose

Na pasta `data-platform`, inicie a API, o frontend e o PostgreSQL:

```bash
docker compose up -d --build postgres credit-api credit-frontend
```

Acessos:

- Swagger da API: `http://localhost:8000/docs`;
- verificação da API: `http://localhost:8000/health`;
- interface Streamlit: `http://localhost:8501`.

Para acompanhar os logs:

```bash
docker compose logs -f credit-api credit-frontend
```

Para interromper os serviços:

```bash
docker compose stop credit-api credit-frontend
```

### Execução local

Com o PostgreSQL disponível:

```bash
export DATABASE_URL="postgresql+psycopg2://airflow:airflow@localhost:5432/data"
.venv/bin/python -m uvicorn MLOps.app.api.main:app --reload
```

Em outro terminal:

```bash
export CREDIT_API_URL="http://localhost:8000"
.venv/bin/python -m streamlit run MLOps/app/frontend/app.py
```

### Endpoints

- `GET /health`: verifica se o modelo foi carregado;
- `GET /model/features`: lista as features esperadas;
- `POST /predict/features`: recebe todas as features no corpo da requisição;
- `POST /predict/customer/{customer_id}`: consulta o cliente no PostgreSQL e
  constrói as features utilizadas pelo modelo.

Exemplo:

```bash
curl -X POST http://localhost:8000/predict/customer/100002
```

Os limites da política são demonstrativos. O `risk_score` deve ser tratado como
uma pontuação de ordenação até que seja realizada a calibração de probabilidade.

## Testes

```bash
.venv/bin/python -m unittest discover -s MLOps/tests -v
```

Os testes verificam a política de crédito, o serviço de inferência, o script
`predict.py` e a consistência entre configurações e artefato persistido.
