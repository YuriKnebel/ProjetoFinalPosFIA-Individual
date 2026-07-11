# MLOps — serviço de predição

Esta camada operacionaliza o artefato treinado em `Model/artifacts`. A pasta
`app` centraliza as duas formas de entrega do resultado: API FastAPI e interface
Streamlit.

## Arquitetura

```text
PostgreSQL ──> feature_service ──┐
                                ├──> model_service ──> credit_policy ──> FastAPI
JSON com features ──────────────┘                                  │
                                                                   └──> Streamlit
```

- `feature_service` consulta as fontes e reproduz as 32 features do cliente;
- `model_service` carrega o pipeline Pickle e calcula o score;
- `credit_policy` converte o score em recomendação de negócio;
- a FastAPI expõe o serviço e o Streamlit consome seus endpoints.

O modelo fornece uma pontuação de risco; a política fornece a recomendação. Os
limites atuais são demonstrativos e exigem validação de negócio.

## Estrutura

```text
MLOps/
├── app/
│   ├── api/
│   │   ├── main.py             # endpoints e ciclo de vida
│   │   ├── config.py           # variáveis de ambiente
│   │   ├── model_service.py    # carregamento e inferência
│   │   ├── feature_service.py  # consulta e transformação das features
│   │   ├── credit_policy.py    # regras de recomendação
│   │   ├── schemas.py          # contratos da API
│   │   └── requirements.txt
│   └── frontend/
│       ├── app.py              # interface Streamlit
│       ├── field_config.py     # campos do formulário
│       └── requirements.txt
├── tests/
├── Dockerfile.api
├── Dockerfile.frontend
└── test-requirements.txt
```

O `docker-compose.yml` permanece na raiz de `data-platform`, pois também
orquestra PostgreSQL, Airflow, Jupyter e Metabase. A DAG que coordena a limpeza
e a construção da ABT fica em `airflow/dags/pipeline_orchestration.py`.

## Execução com Docker Compose

Na pasta `data-platform`:

```bash
docker compose up -d --build postgres credit-api credit-frontend
```

- Swagger: `http://localhost:8000/docs`;
- health check: `http://localhost:8000/health`;
- Streamlit: `http://localhost:8501`.

## Execução local

```bash
.venv/bin/python -m pip install -r MLOps/test-requirements.txt
.venv/bin/python -m uvicorn MLOps.app.api.main:app --reload
```

Em outro terminal:

```bash
.venv/bin/python -m pip install -r MLOps/app/frontend/requirements.txt
.venv/bin/python -m streamlit run MLOps/app/frontend/app.py
```

## Endpoints

- `GET /health`;
- `GET /model/features`;
- `POST /predict/features`;
- `POST /predict/customer/{customer_id}`.

## Testes

```bash
.venv/bin/python -m unittest discover -s MLOps/tests -v
```

## Próximos passos

1. Calibrar o score como probabilidade e definir limites com custos reais.
2. Monitorar drift de dados, performance, latência e disponibilidade.
3. Versionar artefato, configuração e métricas em um registry de modelos.
4. Adicionar autenticação, rastreabilidade e auditoria das decisões.
5. Automatizar testes, build e implantação por CI/CD.
