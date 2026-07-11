# DataPipeline

Esta pasta contém a lógica de ingestão, limpeza, agregação e construção da Analytical Base Table (ABT) do projeto de risco de crédito.

## Contexto do problema de dados

As fontes do Home Credit possuem granularidades diferentes. `application_train` tem uma linha por cliente, enquanto bureau, propostas anteriores e parcelas podem conter muitos registros para o mesmo `sk_id_curr`. Um join direto duplicaria clientes, distorceria o target e daria peso indevido a quem possui mais histórico.

O DataPipeline resolve esse problema em duas etapas: primeiro preserva e trata cada fonte; depois transforma os históricos um-para-muitos em indicadores agregados por cliente. A ABT final mantém a granularidade necessária ao aprendizado supervisionado: uma linha, um target e um conjunto consistente de features por cliente.

Além de preparar dados tecnicamente válidos, o componente busca preservar significado de negócio. Ausência de histórico é representada por flags próprias, anomalias cadastrais são isoladas e variáveis financeiras são tratadas sem apagar diferenças relevantes de risco.

## Responsabilidade

- carregar os CSVs de origem no PostgreSQL;
- criar índices para as etapas de transformação;
- limpar e padronizar as fontes de dados;
- agregar históricos por cliente;
- materializar a tabela `application_abt` com uma linha por cliente;
- documentar a exploração dos dados brutos e tratados.

## Por que ELT no PostgreSQL

Os arquivos de bureau, propostas e parcelas possuem centenas de megabytes. Transferir todas essas tabelas para a memória de um processo Python apenas para agregá-las aumentaria consumo, tempo de execução e risco de falha. A implementação adota ELT:

1. Python controla conexão, parâmetros, arquivos e ciclo das tarefas.
2. Os dados são carregados no PostgreSQL.
3. Limpeza, percentis, agregações e joins são executados em SQL pelo banco.

Essa escolha aproveita o otimizador do PostgreSQL, reduz movimentação de dados e mantém as transformações observáveis como tabelas intermediárias.

## Fluxo

```text
CSVs
  → tabelas brutas no PostgreSQL
  → tabelas *_clean
  → agregações de previous_application, bureau e installments
  → application_abt
  → treinamento do modelo
```

A execução completa é coordenada pela DAG descrita no [README do Airflow](../airflow/README.md).

## Arquivos principais

| Arquivo | Finalidade |
|---|---|
| [`ingestion.py`](./ingestion.py) | Carrega os CSVs em blocos no PostgreSQL. |
| [`ingestion_index.py`](./ingestion_index.py) | Cria índices nas tabelas de origem. |
| [`data_sanitization.py`](./data_sanitization.py) | Aplica limpeza e padronização por SQL. |
| [`data_sanitization_index.py`](./data_sanitization_index.py) | Recria índices nas tabelas tratadas. |
| [`abt_transform.py`](./abt_transform.py) | Agrega históricos e constrói a ABT. |
| [`utils.py`](./utils.py) | Centraliza conexões, carga e utilitários compartilhados. |
| [`config_pipeline.json`](./config_pipeline.json) | Define fontes, tabelas, chunks, índices e parâmetros de limpeza. |
| [`abt_fields.txt`](./abt_fields.txt) | Inventário textual dos campos da ABT. |
| [`df_correlations_with_target.txt`](./df_correlations_with_target.txt) | Registro auxiliar das correlações com o alvo. |

## Implementação da ingestão

[`ingestion.py`](./ingestion.py) recebe uma configuração de tabela por tarefa Airflow e executa:

1. validação de que a tabela pertence ao escopo de `config_pipeline.json`;
2. localização do CSV por nome normalizado;
3. tentativa de leitura UTF-8, com fallback para Latin-1;
4. leitura iterativa com `chunksize` específico para cada fonte;
5. inferência inicial de tipos Pandas → PostgreSQL;
6. criação da tabela no primeiro chunk;
7. append dos blocos por `COPY FROM STDIN` com delimitador tab;
8. log de volumetria após as cargas.

O uso de `COPY` evita inserts linha a linha. Os chunks controlam a memória e permitem tamanhos distintos de lote conforme o volume de cada fonte.

| Fonte | Chunk configurado | Motivo funcional |
|---|---:|---|
| `application_train` | 150.000 | Cadastro largo, com muitas colunas. |
| `previous_application` | 300.000 | Histórico com múltiplas propostas por cliente. |
| `bureau` | 150.000 | Histórico externo com valores e categorias. |
| `installments_payments` | 600.000 | Fonte longa, processada com lote maior. |

## Implementação da limpeza

[`data_sanitization.py`](./data_sanitization.py) cria novas tabelas em vez de sobrescrever as fontes brutas.

### `application_train → application_clean`

Uma CTE calcula estatísticas globais uma única vez e as aplica a todos os clientes:

- medianas dos três scores externos e de `ext_source_mean`;
- medianas de telefone, família, anuidade e renda;
- percentil configurável da renda para winsorização;
- mediana da idade do carro apenas entre clientes que possuem veículo;
- categorias de organização e renda com frequência mínima configurada.

As principais regras são:

| Tema | Tratamento |
|---|---|
| Scores externos | Nulos preenchidos pela mediana e média consolidada em `ext_source_mean`. |
| Renda | Zero tratado como ausência, imputação pela mediana e limite superior no percentil 99. |
| Veículo | `has_car` binário; idade do carro é zero sem carro e mediana quando há carro sem idade informada. |
| Categorias raras | Organização e tipo de renda abaixo da frequência mínima viram `Other_low_freq`. |
| Ausências categóricas | Ocupação e educação recebem `Unknown`. |
| Gênero inválido | `XNA` é convertido em `Unknown`. |
| Idade | `days_birth` negativo é convertido para anos positivos. |
| Emprego | O sentinel `365243` vira `years_employed = 0` e ativa `days_employed_anom`. |
| Indicadores binários e contagens | Ausências selecionadas são preenchidas com zero. |

### Históricos

- `previous_application_clean` normaliza o status do contrato e impede valores negativos de aplicação;
- `bureau_clean` padroniza textos e converte campos monetários e de atraso para tipos numéricos com zero quando adequado;
- `installments_clean` mantém as colunas necessárias e remove linhas sem chaves, vencimento ou valor de parcela.

As tabelas tratadas recebem índices novamente porque são recriadas a cada execução.

## Implementação das agregações

[`abt_transform.py`](./abt_transform.py) reduz cada histórico para uma linha por cliente antes do join final.

| Histórico | Feature agregada | Cálculo resumido |
|---|---|---|
| Propostas anteriores | `prev_refused_rate` | Proporção de propostas com status `Refused`. |
| Bureau | `bureau_avg_days_credit` | Média da antiguidade dos créditos. |
| Bureau | `bureau_last_days_credit` | Crédito mais recente observado. |
| Bureau | `bureau_active_rate` e `bureau_active_count` | Proporção e quantidade de créditos ativos. |
| Bureau | `bureau_closed_rate` | Proporção de créditos encerrados. |
| Bureau | `bureau_debt_credit_ratio` | Dívida total dividida pelo crédito total. |
| Bureau | `bureau_overdue_count` | Quantidade de créditos com atraso. |
| Parcelas | `inst_late_payment_rate` | Proporção de parcelas pagas depois do vencimento. |

Cada agregado é salvo temporariamente e indexado por `sk_id_curr`. Depois do join, as tabelas temporárias são removidas.

## Construção da ABT

A tabela `application_clean` é o lado esquerdo dos joins. Isso preserva todos os clientes do cadastro, mesmo quando não possuem histórico nas demais fontes.

Além das agregações, a etapa final cria:

- `fe_credit_income_percent`: crédito solicitado dividido pela renda;
- `fe_annuity_income_percent`: anuidade dividida pela renda;
- `has_prev_app`, `has_bureau` e `has_installments_history`.

As flags diferenciam ausência de histórico de um histórico cujo indicador agregado vale zero. Os valores agregados ausentes são preenchidos com zero, enquanto a flag preserva a informação de que não houve observação.

O resultado é `application_abt`, contendo identificador, target e as 42 features definidas em [`Model/config_model.json`](../Model/config_model.json).

## Propriedades esperadas da ABT

- exatamente uma linha por `sk_id_curr`;
- target preservado da aplicação principal;
- ausência de duplicação causada pelos históricos;
- nomes e tipos compatíveis com a configuração do modelo;
- flags de presença coerentes com os agregados;
- razões financeiras protegidas contra divisão por zero;
- categorias já reduzidas e padronizadas.

O notebook [`exp_analysis_abt.ipynb`](./exp_analysis_abt.ipynb) verifica qualidade, duplicidade, nulos, constantes, força preditiva e multicolinearidade após a materialização.

## Fontes ingeridas

Os arquivos devem ser colocados em `data-platform/airflow/data/csv`:

- `application_train.csv`;
- `previous_application.csv`;
- `bureau.csv`;
- `installments_payments.csv`.

O escopo e o tamanho dos blocos são controlados por [`config_pipeline.json`](./config_pipeline.json).

## Configuração do pipeline

| Seção | Função |
|---|---|
| `ingestion_table.using_csv` | Fontes autorizadas e tamanho de cada chunk. |
| `database` | Nomes das tabelas brutas, tratadas e da ABT. |
| `indexes` | Chaves declaradas para apoio à indexação. |
| `sanitization.cardinalidade_min_freq` | Frequência mínima antes de agrupar categorias raras. |
| `sanitization.income_winsor_q` | Quantil máximo aplicado à renda. |

O Airflow lê essa configuração no carregamento da DAG e distribui os parâmetros às tarefas. Alterar nomes de tabela ou regras de sanitização deve ser coordenado com a DAG, notebooks e configuração do modelo.

## Notebooks

- [`exp_analysis_raw.ipynb`](./exp_analysis_raw.ipynb): investiga a base bruta, desbalanceamento, ausência de dados, anomalias e oportunidades de engenharia de atributos.
- [`exp_analysis_abt.ipynb`](./exp_analysis_abt.ipynb): valida a ABT final e analisa qualidade, poder preditivo, históricos agregados, multicolinearidade e atributos sensíveis.

Os notebooks podem ser abertos pelo ambiente descrito em [Jupyter](../jupyter/README.md).

## Execução

O caminho recomendado é iniciar PostgreSQL e Airflow e disparar a DAG `pipeline_orchestration`:

```bash
cd data-platform
docker compose up -d --build postgres airflow-init airflow-webserver airflow-scheduler
```

Depois, acesse http://localhost:8080, localize `pipeline_orchestration` e inicie uma execução manual.

## Saídas

- tabelas brutas no banco `data`;
- tabelas tratadas com sufixo `_clean`;
- agregações temporárias por cliente;
- ABT `application_abt`;
- arquivo local [`abt.csv`](./abt.csv), quando materializado pelos fluxos de análise.

## Observabilidade do processamento

As funções registram no log:

- tabela e etapa em processamento;
- quantidade de registros na entrada e saída;
- número do chunk durante a ingestão;
- criação de índices e agregações;
- conclusão ou rollback em caso de erro.

Esses eventos aparecem nos logs das tarefas do Airflow e permitem localizar quedas inesperadas de volumetria entre as camadas.

## Componentes relacionados

- [PostgreSQL](../postgres/README.md)
- [Airflow](../airflow/README.md)
- [Jupyter](../jupyter/README.md)
- [Modelo](../Model/README.md)
