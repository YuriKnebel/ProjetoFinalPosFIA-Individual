# MLOps

Esta pasta operacionaliza o modelo treinado por meio de uma API FastAPI e de uma interface Streamlit. A política de crédito permanece separada do score do modelo.

## Contexto e valor do componente

Um notebook ou arquivo Pickle, isoladamente, não permite que um analista utilize o modelo de forma consistente. A camada MLOps transforma o resultado da modelagem em um serviço com contrato explícito, validação das entradas, política configurável e interface de demonstração.

O desenho resolve quatro preocupações:

- **consistência:** toda predição passa pelo mesmo serviço de modelo;
- **separação de decisão:** score, classe estatística e recomendação de negócio são conceitos distintos;
- **reuso:** frontend, scripts e outros consumidores podem usar a mesma API;
- **demonstração auditável:** resposta, thresholds e versão da política são apresentados juntos.

A implementação é deliberadamente acadêmica: demonstra o serving do modelo e a integração entre componentes. Autenticação, registry, persistência estruturada de auditoria e monitoramento produtivo permanecem evoluções futuras.

## Arquitetura do serviço

```text
Formulário com 42 features ───────────┐
                                     ├→ PredictionService → CreditPolicy → FastAPI
PostgreSQL / application_abt → FeatureService                         │
                                                                      └→ Streamlit
```

- `FeatureService` recupera da ABT as mesmas 42 features usadas no treinamento;
- `PredictionService` carrega o artefato LightGBM e calcula o score;
- `CreditPolicy` converte o score em `approve`, `manual_review` ou `reject`;
- FastAPI expõe os contratos;
- Streamlit oferece preenchimento manual, recuperação editável e consulta direta de clientes.

O score é uma pontuação de ordenação de risco, não uma probabilidade calibrada de inadimplência.

## Responsabilidades e limites

| Componente | Responsabilidade | Não é responsabilidade |
|---|---|---|
| `feature_service` | Recuperar uma linha da ABT e preparar suas features. | Reexecutar a engenharia de atributos sobre as fontes brutas. |
| `model_service` | Validar o artefato, alinhar tipos e calcular score/classe. | Definir aprovação ou rejeição de negócio. |
| `credit_policy` | Traduzir faixas de score em recomendação demonstrativa. | Retreinar ou calibrar o modelo. |
| FastAPI | Gerenciar ciclo de vida, contratos e erros HTTP. | Armazenar histórico definitivo das decisões. |
| Streamlit | Oferecer jornadas de demonstração e explicar o resultado. | Conter o modelo ou acessar diretamente o Pickle. |

Essa separação evita acoplar mudanças da política comercial ao treinamento do algoritmo.

## Estrutura

```text
MLOps/
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── feature_service.py
│   │   ├── model_service.py
│   │   ├── credit_policy.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── app.py
│       ├── field_config.py
│       └── requirements.txt
├── tests/
├── Dockerfile.api
├── Dockerfile.frontend
├── test-requirements.txt
└── README.md
```

## Configuração

| Variável | Finalidade | Padrão no Compose |
|---|---|---|
| `MODEL_PATH` | Caminho do artefato LightGBM | `/app/Model/artifacts/lightgbm_abt.pkl` |
| `DATABASE_URL` | Conexão com o banco `data` | PostgreSQL do Compose |
| `CREDIT_APPROVE_MAX_SCORE` | Limite superior para aprovação | `0.35` |
| `CREDIT_MANUAL_REVIEW_MAX_SCORE` | Limite superior para revisão manual | `0.65` |
| `CREDIT_POLICY_VERSION` | Identificador da política | `demo-v1` |
| `CREDIT_API_URL` | URL consumida pelo frontend | `http://credit-api:8000` |

Os limites são demonstrativos e devem ser validados com custos e regras reais do negócio.

## Implementações críticas

### Carregamento do modelo

No startup da FastAPI, o `lifespan`:

1. valida os limites da política;
2. cria o `PredictionService` com `MODEL_PATH`;
3. carrega e valida o dicionário Pickle;
4. cria o engine SQLAlchemy com `pool_pre_ping=True`;
5. instancia serviço de features e política;
6. registra os serviços em `app.state` para reuso pelas requisições;
7. libera o pool de conexões no shutdown.

O artefato precisa conter modelo, threshold, métricas e uma lista de features. Para compatibilidade, o serviço aceita a chave atual `features` ou a chave histórica `input_features`.

### Preparação das features para inferência

Antes do `predict_proba`, o `PredictionService`:

- rejeita requisições que não contenham todas as features obrigatórias;
- reorganiza as colunas exatamente na ordem do treinamento;
- ignora campos extras durante o reindex;
- restaura `pandas.Categorical` com as categorias salvas no artefato;
- converte as demais features para tipo numérico;
- calcula `risk_score` a partir da classe positiva;
- compara o score com o threshold persistido para gerar `predicted_class`.

Restaurar categorias é essencial para o LightGBM com categóricas nativas: o mesmo texto precisa ocupar a mesma categoria lógica usada durante o ajuste.

### Recuperação do cliente

O `CustomerFeatureService` consulta diretamente `application_abt` por `sk_id_curr`. Identificador e target são removidos antes do retorno. O serviço garante ainda a presença das features de parcelas por compatibilidade com ABTs materializadas anteriormente.

Consumir a ABT evita duplicar na API as regras complexas de agregação do pipeline. A desvantagem consciente é que uma predição por cliente depende da atualização prévia da ABT.

### Política de crédito

O `CreditPolicy` recebe dois limites validados:

```text
score < approve_max_score
  → approve

approve_max_score ≤ score < manual_review_max_score
  → manual_review

score ≥ manual_review_max_score
  → reject
```

A resposta inclui limites e `policy_version`, tornando explícita a regra que produziu a recomendação. O `predicted_class` continua baseado no threshold do modelo e pode divergir da recomendação, pois atende a outra finalidade.

### Tratamento de erros

| Situação | Resposta |
|---|---|
| Cliente inexistente na ABT | HTTP `404`. |
| Falha ao consultar PostgreSQL | HTTP `503`. |
| Features obrigatórias ausentes | HTTP `422` com lista das ausências. |
| Artefato ausente ou inválido no startup | API não conclui a inicialização. |

As requisições de predição são registradas em JSON no stdout do container para apoiar demonstração e diagnóstico. Esse registro não substitui uma trilha de auditoria persistente.

## Inicialização com Docker

Na pasta [`data-platform`](../README.md):

```bash
docker compose up -d --build postgres credit-api credit-frontend
```

Para acompanhar os serviços:

```bash
docker compose logs -f credit-api credit-frontend
```

## URLs

| Serviço | URL |
|---|---|
| Documentação Swagger | http://localhost:8000/docs |
| Health check da API | http://localhost:8000/health |
| Streamlit | http://localhost:8501 |

## Endpoints

| Método e caminho | Finalidade |
|---|---|
| `GET /health` | Informa disponibilidade e carregamento do modelo. |
| `GET /model/features` | Lista as features esperadas pelo modelo. |
| `GET /customers/{customer_id}/features` | Recupera as features de um cliente para edição. |
| `POST /predict/features` | Calcula o score a partir das features fornecidas. |
| `POST /predict/customer/{customer_id}` | Recupera o cliente na ABT e calcula o score. |

### Estrutura da requisição por features

O endpoint recebe um objeto `features` com todas as 42 entradas listadas por `GET /model/features`:

```json
{
  "features": {
    "ext_source_1": 0.50,
    "ext_source_2": 0.62,
    "ext_source_3": 0.48,
    "ext_source_mean": 0.53,
    "age": 35.0,
    "occupation_type": "Laborers"
  }
}
```

O exemplo é abreviado para leitura; uma chamada válida deve incluir todas as features retornadas pelo endpoint de metadados.

### Estrutura da resposta

```json
{
  "source": "provided_features",
  "customer_id": null,
  "risk_score": 0.42,
  "predicted_class": 0,
  "model_decision_threshold": 0.5,
  "policy": {
    "recommendation": "manual_review",
    "reason": "Score na faixa intermediária; requer análise humana.",
    "policy_version": "demo-v1",
    "approve_max_score": 0.35,
    "manual_review_max_score": 0.65
  }
}
```

`source` informa se a pontuação veio do formulário ou do banco. Quando a consulta parte de um cliente armazenado, `customer_id` permite associar o resultado à origem.

## Jornadas do frontend

O Streamlit implementa três formas de demonstração:

### Preencher todos os dados

Renderiza as 42 features agrupadas por contexto. Campos categóricos usam opções controladas, flags usam seleção binária e valores numéricos respeitam limites e passos definidos em `field_config.py`.

### Buscar cliente e editar

Recupera as features com `GET /customers/{id}/features`, mantém o cliente no `session_state`, preenche um novo formulário e permite simular mudanças antes da predição. Essa jornada evidencia como alterações cadastrais ou financeiras afetam o score sem modificar a ABT.

### Consultar cliente do banco

Envia apenas o identificador para `POST /predict/customer/{id}`. A API recupera a ABT e calcula a recomendação sem edição manual.

Em todas as jornadas, o frontend exibe score, classe, origem, threshold do modelo, limites da política, justificativa e resposta JSON completa. Uma mensagem fixa reforça que o score não é probabilidade calibrada.

## Empacotamento

### API

`Dockerfile.api` instala somente as dependências da API, copia `MLOps` e os artefatos de `Model/artifacts`, define `MODEL_PATH` e inicia Uvicorn na porta 8000.

### Frontend

`Dockerfile.frontend` instala Streamlit e Requests, copia a aplicação e inicia o servidor na porta 8501. A comunicação interna usa o DNS do Compose: `http://credit-api:8000`.

Como o código é copiado durante o build, alterações locais exigem reconstrução da imagem correspondente.

## Execução local

Com PostgreSQL e artefato disponíveis:

```bash
cd data-platform
python3 -m venv MLOps/.venv
MLOps/.venv/bin/python -m pip install -r MLOps/app/api/requirements.txt
MLOps/.venv/bin/python -m uvicorn MLOps.app.api.main:app --reload
```

Em outro terminal:

```bash
cd data-platform
MLOps/.venv/bin/python -m pip install -r MLOps/app/frontend/requirements.txt
CREDIT_API_URL=http://localhost:8000 \
  MLOps/.venv/bin/python -m streamlit run MLOps/app/frontend/app.py
```

## Testes

```bash
cd data-platform
MLOps/.venv/bin/python -m pip install -r MLOps/test-requirements.txt
MLOps/.venv/bin/python -m pip install -r MLOps/app/frontend/requirements.txt
MLOps/.venv/bin/python -m unittest discover -s MLOps/tests -v
```

### Cobertura dos testes existentes

| Arquivo | Responsabilidade validada |
|---|---|
| `test_credit_policy.py` | Faixas de aprovação, revisão, rejeição e limites inválidos. |
| `test_model_service.py` | Score válido e rejeição de features ausentes. |
| `test_predict.py` | Inferência pelo script local e contrato do resultado. |
| `test_frontend.py` | Inicialização da aplicação Streamlit. |
| `test_configuration.py` | Estrutura esperada e coerência entre configuração e artefato. |

## Limitações conhecidas

- a política usa limites demonstrativos;
- o score não está calibrado como probabilidade;
- não há autenticação ou autorização nos endpoints;
- requisições e respostas não são persistidas em armazenamento de auditoria;
- a API depende da disponibilidade da ABT no PostgreSQL;
- o artefato é empacotado na imagem e não obtido de um model registry;
- não há monitoramento contínuo de drift, latência ou performance pós-deploy.

## Componentes relacionados

- [Modelo](../Model/README.md)
- [PostgreSQL](../postgres/README.md)
- [Airflow](../airflow/README.md)
- [Arquitetura da plataforma](../README.md)
