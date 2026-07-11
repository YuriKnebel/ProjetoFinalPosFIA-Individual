# Model

Esta pasta reúne a seleção, o treinamento, a avaliação e a inferência local do modelo de risco de crédito.

## Contexto de modelagem

O target do projeto identifica clientes inadimplentes (`target = 1`). A base é fortemente desbalanceada: aproximadamente 8% dos clientes pertencem à classe positiva. Por isso, um classificador pode alcançar alta acurácia prevendo majoritariamente bons pagadores e ainda assim oferecer pouco valor para a política de crédito.

O objetivo da modelagem não é automatizar isoladamente a concessão de crédito. O modelo deve:

- ordenar clientes de acordo com a propensão à inadimplência;
- concentrar maus pagadores nas faixas superiores do score;
- generalizar para clientes não usados no ajuste;
- fornecer drivers interpretáveis e coerentes com o negócio;
- permitir que thresholds sejam avaliados segundo perdas, margem e capacidade operacional;
- manter o mesmo contrato de features no treinamento e na inferência.

Essa finalidade orienta a escolha de ROC AUC, Gini, KS e Average Precision como métricas centrais, complementadas por Brier, matriz de confusão, recall e métricas econômicas de corte.

## Modelo atual

O modelo oficial é um **LightGBM Classifier** com variáveis categóricas nativas e tratamento de desbalanceamento por `class_weight="balanced"`. A saída deve ser interpretada como um **score de propensão à inadimplência**, adequado para ordenação de risco, e não como probabilidade calibrada.

### Por que LightGBM

A seleção não foi definida antecipadamente. Regressão Logística, Random Forest, XGBoost e LightGBM foram comparados sob validação cruzada estratificada e teste externo. O LightGBM foi escolhido por combinar:

- melhor capacidade de discriminação entre as configurações aceitas;
- diferença controlada entre treino, validação e teste;
- suporte eficiente a relações não lineares e interações;
- tratamento nativo das cinco features categóricas;
- bom desempenho em dados tabulares com centenas de milhares de linhas;
- mecanismos de regularização compatíveis com o controle de overfitting.

### Limite de interpretação do score

O método `predict_proba` produz um valor entre 0 e 1, mas `class_weight="balanced"` altera o peso relativo das classes no ajuste. A avaliação também identifica erro de calibração. Portanto:

- o valor é adequado para ranking, decis e políticas de corte;
- valores maiores representam maior propensão à inadimplência;
- `0.70` não deve ser comunicado como “70% de probabilidade real de default”;
- calibração adicional é necessária caso o uso exija probabilidade observável.

O frontend e a documentação da API adotam o termo `risk_score` para preservar essa distinção.

## Dados de entrada

O treinamento lê `application_abt` no PostgreSQL. A tabela possui uma linha por `sk_id_curr`, target e 42 features distribuídas entre:

| Família | Exemplos | Informação representada |
|---|---|---|
| Scores externos | `ext_source_1/2/3`, `ext_source_mean` | Sinais externos consolidados de risco. |
| Perfil e estabilidade | `age`, `years_employed`, `days_employed_anom` | Idade, vínculo e anomalias cadastrais. |
| Capacidade financeira | `amt_income_total`, `amt_credit`, `amt_annuity` | Renda e valores da operação. |
| Razões derivadas | `fe_credit_income_percent`, `fe_annuity_income_percent` | Comprometimento relativo da renda. |
| Histórico interno | `prev_refused_rate`, `has_prev_app` | Experiência em propostas anteriores. |
| Bureau | `bureau_*`, `has_bureau` | Atividade, recência, dívida e atraso externos. |
| Parcelas | `inst_late_payment_rate`, `has_installments_history` | Comportamento de pagamento observado. |
| Categóricas | ocupação, organização, renda, educação e gênero | Segmentos cadastrais tratados. |

A lista ordenada completa está em [`config_model.json`](./config_model.json). `sk_id_curr` e `target` nunca entram como variáveis explicativas.

## Ciclo de desenvolvimento

```text
application_abt
  → split externo estratificado 80/20
  → amostra de busca no conjunto de treino
  → RandomizedSearchCV com 5 folds
  → comparação de quatro famílias de modelos
  → filtro de overfitting
  → seleção do LightGBM
  → avaliação no holdout
  → análise de threshold, explicabilidade e fairness
  → configuração oficial
  → treino final com 100% da ABT
  → artefato para inferência
```

O holdout mede generalização e não participa do ajuste final durante a comparação. Depois que a configuração é escolhida e avaliada, `train.py` treina um modelo de avaliação no split e, em seguida, ajusta o artefato final com toda a ABT.

## Configuração

[`config_model.json`](./config_model.json) é a fonte central para reprodução da modelagem.

| Seção | Conteúdo |
|---|---|
| `metadata` | Projeto, versão, algoritmo, origem, tabela e caminho do artefato. |
| `database` | Hosts e parâmetros de conexão dos ambientes local e Docker. |
| `variables` | Identificador, target, features de entrada e categóricas. |
| `parameters.split` | Holdout, estratificação e semente. |
| `parameters.classifier` | Algoritmo e hiperparâmetros do LightGBM. |
| `parameters.inference` | Threshold de classe persistido no artefato. |
| `validation` | Folds, iterações e tamanho da amostra de busca. |
| `model_results` | Justificativa e métricas de referência da seleção. |

### Hiperparâmetros oficiais

| Parâmetro | Valor | Papel principal |
|---|---:|---|
| `n_estimators` | 268 | Quantidade de árvores. |
| `learning_rate` | 0.0778 | Contribuição incremental de cada árvore. |
| `num_leaves` | 45 | Complexidade máxima das folhas. |
| `max_depth` | 4 | Limita a profundidade das árvores. |
| `min_child_samples` | 263 | Evita folhas apoiadas em poucos clientes. |
| `colsample_bytree` | 0.8444 | Amostra features por árvore. |
| `reg_lambda` | 1.0707 | Regularização L2. |
| `class_weight` | `balanced` | Compensa o desbalanceamento no ajuste. |

Profundidade baixa, folhas com amostra mínima elevada e regularização reduzem a tendência de memorizar o conjunto de treino.

## Estrutura

| Arquivo ou pasta | Finalidade |
|---|---|
| [`config_model.json`](./config_model.json) | Fonte de configuração das features, hiperparâmetros, split, threshold e resultados de referência. |
| [`train.py`](./train.py) | Treina, avalia e persiste o LightGBM. |
| [`predict.py`](./predict.py) | Executa inferência local para um cliente da ABT. |
| [`validacao_modelos.ipynb`](./validacao_modelos.ipynb) | Compara algoritmos e configurações, controla overfitting e seleciona o modelo. |
| [`evaluation.ipynb`](./evaluation.ipynb) | Avalia desempenho, threshold, explicabilidade, fairness e monitoramento. |
| [`requirements.txt`](./requirements.txt) | Dependências da modelagem. |
| [`artifacts/`](./artifacts/) | Modelos persistidos, métricas e resultados de comparação. |

## Notebooks

### [`validacao_modelos.ipynb`](./validacao_modelos.ipynb)

Compara Regressão Logística, Random Forest, XGBoost e LightGBM usando validação cruzada estratificada, busca de hiperparâmetros e comparação entre treino, validação interna e teste externo.

Principais perguntas respondidas:

- qual família de algoritmo oferece o melhor ranking de risco;
- quais configurações apresentam overfitting aceitável;
- se o resultado interno se mantém no holdout;
- quais hiperparâmetros devem alimentar o treinamento oficial.

### [`evaluation.ipynb`](./evaluation.ipynb)

Avalia exclusivamente o modelo selecionado. Inclui métricas de crédito, curvas, decis, análise econômica de threshold, importância por permutação, SHAP, fairness e proposta de monitoramento.

Principais perguntas respondidas:

- qual a capacidade de discriminação no teste externo;
- onde os inadimplentes se concentram ao longo dos decis;
- como diferentes thresholds alteram aprovação, captura e valor esperado;
- quais features movem o score e em qual direção;
- se há diferenças relevantes de desempenho e decisão entre subgrupos;
- quais métricas devem ser acompanhadas após o deploy.

Os notebooks podem ser executados pelo ambiente descrito em [Jupyter](../jupyter/README.md).

## Preparação do ambiente local

Na pasta `data-platform`:

```bash
python3 -m venv Model/.venv
Model/.venv/bin/python -m pip install -r Model/requirements.txt
```

O PostgreSQL deve estar disponível e a ABT `application_abt` deve ter sido criada pelo [pipeline de dados](../DataPipeline/README.md).

## Treinamento

```bash
cd data-platform
PYTHONPATH=DataPipeline Model/.venv/bin/python Model/train.py
```

Treinamento reduzido para validação rápida:

```bash
PYTHONPATH=DataPipeline Model/.venv/bin/python Model/train.py \
  --sample-size 5000 \
  --output /tmp/lightgbm_abt_smoke.pkl
```

O treinamento oficial também é a última etapa da DAG `pipeline_orchestration`.

### O que `train.py` executa

1. Carrega e valida as seções obrigatórias da configuração.
2. Consulta a ABT no PostgreSQL.
3. Seleciona as 42 features e converte as categóricas.
4. Cria holdout estratificado para avaliação da configuração.
5. Treina o modelo de avaliação e calcula AUC, Gini, KS, Average Precision e Brier.
6. Gera relatório de classificação no threshold configurado.
7. Retreina o LightGBM final com toda a ABT.
8. Persiste modelo, features, categorias, métricas e metadados.

O parâmetro `--sample-size` limita a consulta e existe para smoke tests. Ele não deve ser usado para gerar o artefato oficial.

## Inferência local

```bash
cd data-platform
Model/.venv/bin/python Model/predict.py --sk-id 100002
```

O comando consulta o cliente em `application_abt`, carrega `artifacts/lightgbm_abt.pkl` e apresenta score, threshold e decisão de classe.

## Artefatos

| Artefato | Finalidade |
|---|---|
| `artifacts/lightgbm_abt.pkl` | Modelo LightGBM oficial e metadados necessários à inferência. |
| [`artifacts/metrics.json`](./artifacts/metrics.json) | Métricas e hiperparâmetros da execução persistida. |
| [`artifacts/model_comparison.csv`](./artifacts/model_comparison.csv) | Resultado histórico de comparação de modelos. |
| `artifacts/logistic_regression_abt.pkl` | Artefato histórico anterior à seleção do LightGBM. |

### Contrato do artefato LightGBM

O Pickle oficial é um dicionário com os elementos necessários para que outro processo reproduza a inferência:

| Chave | Conteúdo |
|---|---|
| `model` | Instância treinada do `LGBMClassifier`. |
| `features` | Lista ordenada das features esperadas. |
| `categorical_features` | Features que precisam manter dtype categórico. |
| `categories` | Categorias conhecidas durante o treinamento. |
| `decision_threshold` | Corte usado para produzir `predicted_class`. |
| `metrics` | Métricas do holdout do modelo de avaliação. |
| `algorithm` | Identificação do algoritmo. |
| `hyperparameters` | Configuração utilizada no ajuste. |
| `trained_at_utc` | Data e hora do treinamento. |
| `config_version` | Versão lógica da configuração. |

A API aceita `features` e normaliza internamente esse nome para `input_features`, preservando compatibilidade com artefatos anteriores.

## Resultados do artefato atual

O arquivo [`metrics.json`](./artifacts/metrics.json) registra:

| Métrica | Valor | Leitura |
|---|---:|---|
| ROC AUC | 0.7650 | Capacidade geral de ordenar bons e maus clientes. |
| Gini | 0.5300 | Transformação do AUC usada em crédito. |
| KS | 0.4051 | Separação máxima acumulada entre as classes. |
| Average Precision | 0.2563 | Qualidade do ranking da classe rara. |
| Brier | 0.1909 | Erro probabilístico; reforça a necessidade de calibração. |

Essas métricas descrevem a execução persistida em `metrics.json`. Resultados de notebooks podem representar outros estágios de seleção ou execuções anteriores e devem ser interpretados no contexto indicado por cada análise.

## Thresholds e política de crédito

Há dois conceitos diferentes:

- **threshold do modelo:** persistido no artefato e usado para `predicted_class`;
- **limites da política:** configurados na API e usados para recomendar aprovação, revisão ou rejeição.

O notebook de avaliação explora thresholds estatísticos e econômicos. Os limites da API são demonstrativos e não representam uma política final validada com custos reais.

## Reprodutibilidade e compatibilidade

- use a mesma versão de LightGBM e dependências registrada em [`requirements.txt`](./requirements.txt);
- não altere a ordem ou o tipo das features sem retreinar;
- mudanças na ABT devem ser refletidas em `config_model.json`;
- Pickle deve ser carregado apenas de origem confiável;
- alterações de categorias exigem novo artefato para preservar o contrato de inferência.

## Componentes relacionados

- [Pipeline de dados](../DataPipeline/README.md)
- [Airflow](../airflow/README.md)
- [MLOps](../MLOps/README.md)
