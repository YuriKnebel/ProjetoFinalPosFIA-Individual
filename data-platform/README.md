# Plataforma de dados e MLOps para risco de crédito

Esta pasta implementa a arquitetura completa do projeto: ingestão e transformação dos dados, exploração analítica, treinamento do modelo e disponibilização do score por API e interface web.

O caso utiliza os dados da competição [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview) para construir um score de propensão à inadimplência. O score ordena clientes por risco e apoia a política de crédito; ele não substitui regras cadastrais, prevenção a fraude, capacidade de pagamento ou revisão humana.

## Contexto e objetivos da plataforma

Uma decisão de crédito precisa equilibrar dois erros com impactos distintos: aprovar um cliente que se tornará inadimplente e recusar um cliente que pagaria corretamente. Como a classe inadimplente é uma minoria expressiva da base, a acurácia isolada não é suficiente para avaliar a solução. A plataforma foi desenhada para transformar múltiplos históricos relacionais em um sinal de risco reproduzível, interpretável e consumível por uma aplicação.

Os objetivos arquiteturais são:

- **reprodutibilidade:** dados, transformações, features e hiperparâmetros possuem fontes de configuração identificáveis;
- **separação de responsabilidades:** pipeline, modelo, serviço de inferência e política de crédito evoluem em componentes distintos;
- **consistência entre treino e inferência:** API e treinamento utilizam a mesma ABT e o mesmo contrato de features;
- **processamento eficiente:** joins e agregações de grande volume são executados no PostgreSQL;
- **rastreabilidade acadêmica:** notebooks registram exploração, seleção, avaliação, interpretabilidade e critérios de negócio;
- **portabilidade local:** toda a infraestrutura necessária pode ser iniciada com Docker Compose;
- **decisão humana preservada:** o score apoia a política, sem automatizar sozinho a concessão de crédito.

## Escopo funcional

A plataforma cobre dois ciclos complementares:

1. **Ciclo de desenvolvimento e treinamento:** ingestão das fontes, preparação da ABT, análise, comparação de modelos, treinamento e persistência do artefato.
2. **Ciclo de inferência:** recuperação ou fornecimento das features, cálculo do score, aplicação da política demonstrativa e apresentação do resultado.

Monitoramento contínuo, registry de modelos, autenticação e implantação produtiva estão descritos como próximos passos; não fazem parte da implementação atual.

## Arquitetura

```text
Arquivos Home Credit
        │
        ▼
airflow/data/csv
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ Airflow: pipeline_orchestration                              │
│ ingestão → limpeza → agregações → ABT → treinamento          │
└───────────────────────────────┬──────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ PostgreSQL            │
                    │ raw, clean e ABT      │
                    └───────┬─────────┬─────┘
                            │         │
                    ┌───────▼───┐ ┌───▼────────────────┐
                    │ Jupyter   │ │ Modelo LightGBM    │
                    │ EDA       │ │ artefato + métricas│
                    └───────────┘ └─────────┬──────────┘
                                           │
                              ┌────────────▼────────────┐
                              │ FastAPI + política      │
                              └────────────┬────────────┘
                                           │
                                      ┌────▼─────┐
                                      │Streamlit │
                                      └──────────┘
```

### Decisões arquiteturais principais

- **ELT no PostgreSQL:** limpeza, agregações e joins são executados próximos aos dados.
- **Airflow como orquestrador:** a DAG coordena funções mantidas nas pastas de pipeline e modelo.
- **Configuração separada do código:** tabelas, features e hiperparâmetros ficam em arquivos JSON.
- **Artefato único de inferência:** o LightGBM e seus metadados são persistidos para consumo pela API.
- **Modelo e política desacoplados:** o modelo gera score; a política converte faixas em recomendações.
- **Duas formas de consumo:** predição por features fornecidas ou por cliente recuperado da ABT.

## Camadas lógicas

| Camada | Artefatos principais | Papel na solução |
|---|---|---|
| Origem | CSVs do Home Credit | Dados cadastrais e históricos usados pelo estudo. |
| Persistência bruta | `application_train`, `previous_application`, `bureau`, `installments_payments` | Preserva as fontes carregadas antes das regras analíticas. |
| Tratamento | tabelas com sufixo `_clean` | Padroniza categorias, ausências, anomalias e valores financeiros. |
| Agregação | tabelas temporárias por `sk_id_curr` | Converte relações um-para-muitos em features por cliente. |
| Analítica | `application_abt` | Contrato tabular compartilhado entre análise, treinamento e inferência. |
| Modelagem | `config_model.json`, notebooks, `train.py` | Seleciona, avalia e treina o LightGBM. |
| Artefatos | `lightgbm_abt.pkl`, `metrics.json` | Transporta modelo, categorias, features e metadados para inferência. |
| Serving | FastAPI e política de crédito | Expõe o score e converte faixas em recomendações. |
| Experiência | Streamlit | Permite demonstrar preenchimento, recuperação e consulta de clientes. |

## Topologia de execução

| Serviço do Compose | Imagem ou build | Dependência principal | Papel |
|---|---|---|---|
| `postgres` | `postgres:15` | volume `pgdata` | Bancos `airflow` e `data`. |
| `airflow-init` | `airflow/` | PostgreSQL saudável | Migração, usuário, pools e permissões. |
| `airflow-webserver` | `airflow/` | inicialização concluída | Interface e API do Airflow. |
| `airflow-scheduler` | `airflow/` | inicialização concluída | Agendamento e execução das tarefas. |
| `jupyter` | `jupyter/` | PostgreSQL | Ambiente de exploração e modelagem. |
| `credit-api` | `MLOps/Dockerfile.api` | PostgreSQL e artefato | Inferência e política de crédito. |
| `credit-frontend` | `MLOps/Dockerfile.frontend` | API | Interface demonstrativa. |

## Componentes

| Componente | Responsabilidade | Documentação |
|---|---|---|
| PostgreSQL | Persistência das fontes, tabelas tratadas e ABT | [postgres/README.md](./postgres/README.md) |
| Airflow | Orquestração ponta a ponta do pipeline | [airflow/README.md](./airflow/README.md) |
| DataPipeline | Ingestão, limpeza, agregações e ABT | [DataPipeline/README.md](./DataPipeline/README.md) |
| Jupyter | Ambiente dos notebooks de análise e modelagem | [jupyter/README.md](./jupyter/README.md) |
| Model | Seleção, treinamento, avaliação e inferência local | [Model/README.md](./Model/README.md) |
| MLOps | API, política de crédito, frontend e testes | [MLOps/README.md](./MLOps/README.md) |

## Fluxo de dados e modelo

1. Os CSVs são colocados em `airflow/data/csv`.
2. A DAG carrega as quatro fontes no banco `data`.
3. O pipeline cria tabelas tratadas e agregações por `sk_id_curr`.
4. A tabela `application_abt` consolida as features preditoras em uma linha por cliente.
5. O treinamento selecionado gera `Model/artifacts/lightgbm_abt.pkl` e `metrics.json`.
6. A API carrega o artefato e consulta a ABT quando recebe um identificador de cliente.
7. A política transforma o score em aprovação, revisão manual ou rejeição demonstrativa.
8. O Streamlit disponibiliza formulário, recuperação editável e consulta direta.

## Cenário de treinamento ponta a ponta

```text
Analista disponibiliza CSVs
  → Airflow valida o escopo configurado
  → cada fonte é carregada em chunks
  → índices apoiam limpeza e joins
  → fontes são padronizadas em tabelas clean
  → históricos são agregados por cliente
  → ABT é materializada
  → LightGBM é treinado e avaliado em holdout
  → modelo final é retreinado com toda a ABT
  → artefato e métricas são persistidos
```

O pipeline pode ser reexecutado para reconstruir as tabelas derivadas e atualizar o artefato. A seleção de algoritmo e hiperparâmetros permanece documentada nos notebooks, enquanto a DAG executa a configuração já escolhida.

## Cenários de inferência

### Features fornecidas

O consumidor envia as features esperadas (listadas por `GET /model/features`) para `POST /predict/features`. A API alinha tipos e ordem, calcula o score e aplica a política configurada.

### Cliente armazenado

O consumidor informa `sk_id_curr`. O serviço lê `application_abt`, remove identificador e target e envia as mesmas features ao modelo. Há duas modalidades:

- recuperar as features para edição com `GET /customers/{customer_id}/features`;
- calcular diretamente com `POST /predict/customer/{customer_id}`.

Essa abordagem reduz divergência entre engenharia de atributos offline e online, pois a inferência por cliente consome a ABT já materializada.

## Contratos entre componentes

| Contrato | Fonte de verdade | Consumidores |
|---|---|---|
| Fontes e tabelas do pipeline | `DataPipeline/config_pipeline.json` | DAG e módulos de transformação. |
| Features e hiperparâmetros | `Model/config_model.json` | treinamento, avaliação e validações. |
| Artefato de inferência | `Model/artifacts/lightgbm_abt.pkl` | script local e API. |
| Schema HTTP | `MLOps/app/api/schemas.py` | API e frontend. |
| Limites da política | variáveis `CREDIT_*` | API e apresentação do resultado. |

Alterações nas features exigem atualização coordenada da ABT, configuração do modelo, novo treinamento e campos do frontend.

## Estrutura

```text
data-platform/
├── airflow/             # ambiente e DAG de orquestração
├── DataPipeline/        # ingestão, transformações, ABT e EDA
├── jupyter/             # imagem do ambiente de notebooks
├── MLOps/               # FastAPI, Streamlit e testes
├── Model/               # treinamento, avaliação e artefatos
├── postgres/            # inicialização do banco data
├── Dados/               # pasta reservada aos CSVs da entrega acadêmica
├── docker-compose.yml   # composição dos serviços
└── README.md
```

## Pré-requisitos

- Docker com suporte a Compose v2;
- aproximadamente 2 GB livres para os CSVs de origem e volumes locais;
- arquivos da competição Home Credit correspondentes às quatro fontes configuradas.

Para execução local fora de containers, também é necessário Python compatível com as dependências do componente utilizado.

## Configuração inicial

1. Configure o token do Jupyter em `data-platform/.env`:

```dotenv
JUPYTER_TOKEN=defina-um-token-local
```

2. Coloque estes arquivos em `data-platform/airflow/data/csv`:

```text
application_train.csv
previous_application.csv
bureau.csv
installments_payments.csv
```

3. Entre na pasta da plataforma:

```bash
cd data-platform
```

## Inicialização completa

```bash
docker compose up -d --build
```

O `airflow-init` termina após preparar o ambiente; os demais serviços permanecem ativos.

Para acompanhar o estado:

```bash
docker compose ps
docker compose logs -f airflow-scheduler credit-api credit-frontend
```

## Inicialização por camada

Infraestrutura e orquestração:

```bash
docker compose up -d --build postgres airflow-init
docker compose up -d airflow-webserver airflow-scheduler
```

Notebooks:

```bash
docker compose up -d --build jupyter
```

Serviço de predição:

```bash
docker compose up -d --build postgres credit-api credit-frontend
```

## URLs locais

| Componente | URL | Credencial |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| JupyterLab | http://localhost:8888 | `JUPYTER_TOKEN` |
| Swagger da API | http://localhost:8000/docs | — |
| Health check | http://localhost:8000/health | — |
| Streamlit | http://localhost:8501 | — |

## Execução do pipeline

1. Acesse o Airflow em http://localhost:8080.
2. Habilite a DAG `pipeline_orchestration`.
3. Inicie uma execução manual.
4. Acompanhe as tarefas até o treinamento e a persistência do modelo.

Detalhes das tarefas, pools e entradas estão no [README do Airflow](./airflow/README.md).

## Preparação manual dos CSVs de entrega

O componente `DataPipeline` inclui o utilitário [`export_data.py`](./DataPipeline/export_data.py), executado manualmente para preparar os CSVs da entrega acadêmica. Ele exporta tabelas já materializadas no PostgreSQL por meio de `COPY TO STDOUT`, sem carregar a tabela completa na memória.

O utilitário não integra a DAG e não altera o fluxo operacional da plataforma. Sua execução ocorre somente depois que o pipeline tiver criado as tabelas que serão entregues.

As dependências, os pré-requisitos, o comando de execução e a forma de selecionar a tabela de origem estão documentados na seção [Exportação de tabelas para CSV](./DataPipeline/README.md#exportação-de-tabelas-para-csv).

## Notebooks oficiais

| Notebook | Finalidade |
|---|---|
| [`DataPipeline/exp_analysis_raw.ipynb`](./DataPipeline/exp_analysis_raw.ipynb) | Exploração e diagnóstico das fontes brutas. |
| [`DataPipeline/exp_analysis_abt.ipynb`](./DataPipeline/exp_analysis_abt.ipynb) | Validação e exploração da ABT tratada. |
| [`Model/validacao_modelos.ipynb`](./Model/validacao_modelos.ipynb) | Comparação, tuning e controle de overfitting. |
| [`Model/evaluation.ipynb`](./Model/evaluation.ipynb) | Avaliação final, threshold, explicabilidade e fairness. |

## Parada dos serviços

```bash
docker compose stop
```

Para remover apenas os containers e a rede, preservando o volume do PostgreSQL:

```bash
docker compose down
```

## Próximos passos

- calibrar o score quando houver necessidade de interpretação probabilística;
- validar thresholds com custos reais do negócio;
- implementar monitoramento contínuo de dados, modelo e serviço;
- formalizar versionamento, rastreabilidade e auditoria das decisões;
- automatizar testes, build e implantação.
