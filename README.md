# Projeto Final FIA — risco de crédito com IA

Projeto final da pós-graduação em Engenharia de Dados da FIA Labdata. A solução implementa o ciclo de uma aplicação de Machine Learning para risco de crédito: ingestão, qualidade e transformação dos dados, construção da ABT, seleção e avaliação do modelo, orquestração, API e interface web.

Os dados são baseados na competição [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/overview).

## O desafio

Decidir crédito envolve dois custos opostos: a perda causada por clientes inadimplentes e a receita perdida quando bons clientes são recusados. Na base utilizada, somente cerca de 8% dos clientes pertencem à classe inadimplente. Esse desbalanceamento torna inadequado avaliar a solução apenas por acurácia e exige um fluxo que conecte engenharia de dados, modelagem, interpretabilidade e política de decisão.

O projeto foi construído como uma solução integrada, e não apenas como um experimento de notebook. Quatro fontes históricas são carregadas e transformadas, relações um-para-muitos são consolidadas em uma ABT, diferentes algoritmos são comparados e o modelo selecionado é disponibilizado por serviço e interface web.

## Objetivo de negócio

Produzir um score de propensão à inadimplência que ajude a priorizar clientes para aprovação, revisão manual ou rejeição. O modelo fornece um sinal de ordenação de risco; a decisão final permanece sujeita à política de crédito e à análise humana.

## O que foi entregue

- pipeline ELT em PostgreSQL para quatro fontes do Home Credit;
- ABT com uma linha por cliente e 42 features;
- EDA das fontes brutas e da base tratada;
- comparação entre Regressão Logística, Random Forest, XGBoost e LightGBM;
- controle de overfitting com validação cruzada e teste externo;
- LightGBM treinado com categóricas nativas e artefato reproduzível;
- avaliação com AUC, Gini, KS, Average Precision, Brier, decis e thresholds;
- interpretabilidade por permutação e SHAP;
- análise de fairness e proposta de monitoramento;
- orquestração ponta a ponta com Airflow;
- API FastAPI, política de crédito configurável e frontend Streamlit;
- ambientes Docker para banco, Airflow, Jupyter, API e frontend.

## Visão da solução

```text
Home Credit CSVs
  → PostgreSQL
  → Airflow + pipeline ELT
  → ABT por cliente
  → LightGBM
  → FastAPI + política de crédito
  → Streamlit
```

A implementação completa e o desenho arquitetural estão documentados em [data-platform/README.md](./data-platform/README.md).

## Resumo da metodologia utilizada

1. Exploração das fontes e do problema de desbalanceamento.
2. Limpeza, padronização e engenharia de atributos.
3. Consolidação de cadastro, bureau, propostas e parcelas em uma ABT.
4. Comparação entre modelos com validação cruzada e controle de overfitting.
5. Seleção de um LightGBM com categóricas nativas.
6. Avaliação em holdout, análise econômica de threshold e interpretabilidade.
7. Persistência do artefato e disponibilização por API e interface web.

O score retornado pelo modelo deve ser tratado como uma pontuação de propensão para ordenação de risco, não como probabilidade calibrada.

## Resultados principais

O artefato LightGBM atual registra:

| Métrica | Resultado |
|---|---:|
| ROC AUC | 0.7650 |
| Gini | 0.5300 |
| KS | 0.4051 |
| Average Precision | 0.2563 |
| Brier | 0.1909 |

Os resultados mostram poder de ordenação útil para triagem de risco. O Brier e a curva de calibração, porém, indicam que o score não deve ser comunicado como probabilidade real sem uma etapa adicional de calibração.

## Implementações críticas

### Engenharia de dados

Os CSVs são lidos em chunks e carregados por `COPY` no PostgreSQL. Limpeza, percentis, agregações e joins são executados em SQL. Antes do join final, bureau, propostas e parcelas são reduzidos a uma linha por cliente, evitando duplicação da aplicação principal.

### Orquestração

A DAG `pipeline_orchestration` usa dynamic task mapping para as quatro fontes, paraleliza limpezas e agregações com pools e só libera o treinamento depois da materialização da ABT.

### Modelagem

O LightGBM foi selecionado após comparação de quatro famílias, busca de hiperparâmetros e filtro de overfitting. As categorias são tratadas nativamente e o modelo final é retreinado com toda a ABT depois da avaliação da configuração.

### Inferência e decisão

A API restaura a ordem, os tipos e as categorias salvas no artefato antes de calcular o score. A política de crédito é separada do modelo e converte faixas configuráveis em aprovação, revisão manual ou rejeição demonstrativa.

Detalhes e justificativas estão nos READMEs de cada componente.

## Estrutura e documentação

| Pasta | Finalidade | README |
|---|---|---|
| `data-platform/` | Arquitetura e operação de toda a plataforma | [data-platform](./data-platform/README.md) |
| `data-platform/airflow/` | Orquestração do pipeline e treinamento | [Airflow](./data-platform/airflow/README.md) |
| `data-platform/DataPipeline/` | Ingestão, limpeza, ABT e EDA | [DataPipeline](./data-platform/DataPipeline/README.md) |
| `data-platform/jupyter/` | Ambiente de notebooks | [Jupyter](./data-platform/jupyter/README.md) |
| `data-platform/Model/` | Seleção, treinamento e avaliação do modelo | [Model](./data-platform/Model/README.md) |
| `data-platform/MLOps/` | FastAPI, Streamlit, política e testes | [MLOps](./data-platform/MLOps/README.md) |
| `data-platform/postgres/` | Inicialização e persistência relacional | [PostgreSQL](./data-platform/postgres/README.md) |

## Execução rápida

1. Coloque os arquivos de origem em `data-platform/airflow/data/csv`.
2. Defina `JUPYTER_TOKEN` em `data-platform/.env`.
3. Inicie a plataforma:

```bash
cd data-platform
docker compose up -d --build
```

4. Acesse o Airflow em http://localhost:8080 com `admin` / `admin`.
5. Habilite e execute manualmente a DAG `pipeline_orchestration`.

## Acessos locais

| Serviço | URL |
|---|---|
| Airflow | http://localhost:8080 |
| JupyterLab | http://localhost:8888 |
| Swagger da API | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

## Notebooks principais

- [`exp_analysis_raw.ipynb`](./data-platform/DataPipeline/exp_analysis_raw.ipynb): análise das fontes brutas.
- [`exp_analysis_abt.ipynb`](./data-platform/DataPipeline/exp_analysis_abt.ipynb): análise da ABT tratada.
- [`validacao_modelos.ipynb`](./data-platform/Model/validacao_modelos.ipynb): comparação e seleção do modelo.
- [`evaluation.ipynb`](./data-platform/Model/evaluation.ipynb): avaliação, threshold, explicabilidade e fairness.

## Treinamento local

Com PostgreSQL e ABT disponíveis:

```bash
cd data-platform
python3 -m venv Model/.venv
Model/.venv/bin/python -m pip install -r Model/requirements.txt
PYTHONPATH=DataPipeline Model/.venv/bin/python Model/train.py
```

Para instruções detalhadas, consulte [Model/README.md](./data-platform/Model/README.md).

## Serviço de predição

```bash
cd data-platform
docker compose up -d --build postgres credit-api credit-frontend
```

Consulte contratos, endpoints, configuração e testes em [MLOps/README.md](./data-platform/MLOps/README.md).
