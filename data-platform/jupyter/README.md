# Jupyter

Esta pasta define o ambiente JupyterLab usado para exploração de dados, validação da ABT e desenvolvimento dos notebooks de modelagem.

## Papel no ciclo analítico

O Jupyter registra as decisões que transformaram exploração em implementação. Os notebooks não são o pipeline produtivo: eles permitem investigar hipóteses, comparar alternativas, visualizar resultados e documentar evidências que depois orientam os scripts e configurações oficiais.

Neste projeto, o ambiente cobre quatro momentos:

1. diagnóstico das fontes brutas;
2. validação da ABT construída pelo pipeline;
3. comparação e seleção do algoritmo;
4. avaliação aprofundada do modelo escolhido.

Separar esses momentos reduz o risco de misturar exploração com transformação operacional e deixa claro qual notebook responde a cada pergunta.

## Responsabilidade

- disponibilizar um ambiente reproduzível para notebooks;
- montar os diretórios `DataPipeline` e `Model` no workspace;
- instalar bibliotecas de análise, modelagem e interpretabilidade.

## Estrutura

```text
jupyter/
├── Dockerfile
├── README.md
└── requirements.txt
```

O [`Dockerfile`](./Dockerfile) estende a imagem `jupyter/datascience-notebook` e instala as dependências descritas em [`requirements.txt`](./requirements.txt).

As dependências adicionais incluem conectividade PostgreSQL, LightGBM, XGBoost, SHAP e scikit-learn. Assim, o mesmo ambiente consegue consultar a ABT, comparar modelos e gerar explicações.

## Inicialização

Na pasta [`data-platform`](../README.md):

```bash
docker compose up -d --build jupyter
```

Para acompanhar a inicialização:

```bash
docker compose logs -f jupyter
```

## Acesso

- JupyterLab: http://localhost:8888
- Token: valor de `JUPYTER_TOKEN` no arquivo `data-platform/.env`

No container, os diretórios são montados em:

| Diretório do projeto | Caminho no Jupyter |
|---|---|
| `DataPipeline/` | `/home/jovyan/work/DataPipeline` |
| `Model/` | `/home/jovyan/work/Model` |

## Notebooks disponíveis

### Pipeline de dados

- [`exp_analysis_raw.ipynb`](../DataPipeline/exp_analysis_raw.ipynb): análise exploratória dos dados brutos, incluindo qualidade, valores ausentes, distribuição do alvo e sinais iniciais.
- [`exp_analysis_abt.ipynb`](../DataPipeline/exp_analysis_abt.ipynb): análise da ABT tratada, incluindo integridade, força preditiva, multicolinearidade e fairness.

### Modelagem

- [`validacao_modelos.ipynb`](../Model/validacao_modelos.ipynb): compara os algoritmos, executa a busca de hiperparâmetros e avalia overfitting.
- [`evaluation.ipynb`](../Model/evaluation.ipynb): avalia o LightGBM selecionado, política de corte, interpretabilidade, fairness e métricas de monitoramento.

## Sequência recomendada

```text
exp_analysis_raw.ipynb
  → pipeline_orchestration materializa application_abt
  → exp_analysis_abt.ipynb
  → validacao_modelos.ipynb
  → config_model.json + train.py
  → evaluation.ipynb
```

### 1. Dados brutos

Use `exp_analysis_raw.ipynb` para compreender desbalanceamento, ausência, anomalias, cardinalidade e relações entre as fontes. As decisões relevantes devem ser refletidas em `data_sanitization.py` e `abt_transform.py`.

### 2. ABT tratada

Use `exp_analysis_abt.ipynb` depois da execução do pipeline. Ele valida granularidade, nulos, duplicidades, redundâncias, features históricas e possíveis atributos sensíveis.

### 3. Seleção do modelo

Use `validacao_modelos.ipynb` para comparar as famílias de algoritmos sob o mesmo split, validação cruzada e critérios de overfitting. O resultado escolhido deve alimentar `config_model.json`.

### 4. Avaliação final

Use `evaluation.ipynb` para avaliar o LightGBM oficial, interpretar drivers, analisar thresholds e registrar limitações, fairness e monitoramento.

## Relação com os componentes operacionais

| Notebook | Implementação que recebe suas decisões |
|---|---|
| EDA bruta | `DataPipeline/data_sanitization.py` e `abt_transform.py` |
| EDA da ABT | `Model/config_model.json` |
| Validação | `Model/config_model.json` e `train.py` |
| Avaliação | política de crédito, apresentação e próximos passos de MLOps |

Os notebooks contêm resultados persistidos de execuções específicas. Para comparar números, verifique se configuração, seed, split e artefato pertencem à mesma execução.

## Componentes relacionados

- [Pipeline de dados](../DataPipeline/README.md)
- [Modelo](../Model/README.md)
