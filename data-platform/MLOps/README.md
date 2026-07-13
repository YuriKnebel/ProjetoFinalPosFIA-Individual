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
Formulário de features ───────────────┐
                                     ├→ PredictionService → CreditPolicy → FastAPI
PostgreSQL / application_abt → FeatureService                         │
                                                                      └→ Streamlit
```

- `FeatureService` recupera da ABT as mesmas features usadas no treinamento;
- `PredictionService` carrega o artefato LightGBM e calcula o score;
- `CreditPolicy` converte o score em `approve`, `manual_review` ou `reject`;
- FastAPI expõe os contratos;
- Streamlit oferece preenchimento manual, recuperação editável e consulta direta de clientes.

O score é uma pontuação de ordenação de risco, não uma probabilidade calibrada de inadimplência.

### Fluxo em camadas

Cada requisição de predição atravessa camadas com **responsabilidade única** — é isso que mantém o modelo isolado da regra de negócio e da apresentação:

```text
Streamlit  (apresentação)
   │  HTTP
   ▼
FastAPI    (contrato / transporte)
   │
   ├─ FeatureService ───→ acesso a dados: recupera as features do cliente na ABT
   ├─ PredictionService → inferência: alinha o contrato do artefato e calcula o score
   └─ CreditPolicy ─────→ regra de negócio: converte o score em recomendação
```

- **acesso a dados** (`feature_service`) e **inferência** (`model_service`) não conhecem regra de negócio;
- **política** (`credit_policy`) não conhece o modelo — recebe apenas um score;
- **transporte** (FastAPI) e **apresentação** (Streamlit) não contêm lógica de crédito.

### Decisões arquiteturais

- **Consistência treino ↔ inferência pela ABT.** O `feature_service` lê a **mesma** `application_abt` usada no treinamento; as features online são idênticas às offline **por construção**. A API **não re-implementa** a engenharia de atributos do pipeline, eliminando *training/serving skew*. Custo consciente: a predição por cliente depende de a ABT estar atualizada.
- **Modelo e política desacoplados.** O modelo entrega um **score de ordenação** (estável, versionado no artefato); a **política de crédito** o traduz em recomendação por **limiares configuráveis**, que mudam sem re-treinar. Por isso `predicted_class` (limiar do modelo) e `recommendation` (política) são conceitos distintos e podem divergir.
- **Contrato dirigido pelo artefato.** A lista de features, as categorias e o threshold viajam dentro do próprio artefato; a API valida e alinha a entrada contra esse contrato antes de pontuar. `schemas.py` formaliza o contrato HTTP e o frontend o consome — uma **fonte de verdade única** que flui de **treino → artefato → API → UI**.
- **Dependências carregadas no startup.** Modelo e engine de banco são criados **uma vez** no `lifespan` e guardados em `app.state`; as requisições os reutilizam, sem recarregar o modelo por chamada. O pool usa `pool_pre_ping` para resiliência a conexões ociosas.
- **Três modos de consumo sobre o mesmo núcleo.** O caminho de predição (`_predict`) é único; muda apenas a **origem das features** — fornecidas pelo consumidor, recuperadas da ABT por `sk_id_curr`, ou recuperadas e **editadas** antes de reavaliar.

### Fluxo do contrato

O mesmo contrato de features atravessa treino, artefato e serviço — nada é redefinido no caminho:

```text
train.py  ──→  artefato .pkl  ──→  PredictionService  ──→  /model/features  ──→  Streamlit / field_config
(features,      (features,          (valida e alinha         (expõe o             (renderiza os
 categorias,     categorias,         a entrada ao             contrato)             mesmos campos)
 threshold)      threshold)          contrato)
```

## Aplicações implementadas

### API FastAPI (`app/api`)

Serviço de scoring que expõe o modelo como serviço de predição:

- **carga resiliente do modelo** — o `lifespan` inicia uma tarefa em segundo plano que tenta carregar e validar o artefato; uma falha é registrada no log e uma nova tentativa ocorre após o intervalo configurado;
- **documentação viva** — OpenAPI/Swagger em `/docs`, gerada a partir dos contratos de `schemas.py`;
- **capacidades** — prontidão (`/health`), metadados de features (`/model/features`), recuperação das features de um cliente (`/customers/{id}/features`) e **dois modos de predição** (por features fornecidas e por cliente armazenado na ABT);
- **validação e erros tipados** — features obrigatórias ausentes → `422` com a lista; cliente inexistente → `404`; falha de banco ou modelo indisponível → `503`;
- **rastreabilidade** — cada predição é registrada em **JSON no stdout** do container (apoio a demonstração e diagnóstico; não substitui auditoria persistente);
- **separação de decisão** — a resposta traz, junto ao score, a recomendação da política e os limiares que a produziram.

### Interface Streamlit (`app/frontend`)

Simulador para o analista de crédito, que **consome a API** e nunca acessa o modelo diretamente:

- **barra lateral** — URL da API configurável e botão **"Verificar conexão"** (checa `/health` e se o modelo está carregado);
- **três abas** — *Preencher todos os dados*, *Buscar cliente e editar* e *Consultar cliente do banco*;
- **formulário dinâmico** — campos agrupados por contexto e gerados a partir de `field_config.py` (categóricos com opções controladas, flags binárias, numéricos com limites e passos);
- **jornada de edição** — carrega as features de um cliente da ABT, permite **ajustar** os campos e reavaliar, evidenciando o efeito de mudanças no score **sem alterar a ABT**;
- **visão do resultado** — faixa de recomendação (cor/ícone), métricas de score, classe prevista e origem, barra de posição na escala de risco, legenda com limiar e política, aviso de que o score **não é probabilidade calibrada** e os **JSONs** enviado e recebido.

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
| `CREDIT_APPROVE_MAX_SCORE` | Limite superior para aprovação | `0.50` |
| `CREDIT_MANUAL_REVIEW_MAX_SCORE` | Limite superior para revisão manual | `0.60` |
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
| Artefato ausente ou inválido | A API permanece ativa, registra o erro, repete a carga periodicamente e responde HTTP `503` nos endpoints que dependem do modelo. |

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
| `GET /health` | Verifica se a API está pronta para predição: retorna `200` com o modelo carregado ou `503` enquanto a carga não for concluída. |
| `GET /model/features` | Lista as features esperadas pelo modelo. |
| `GET /customers/{customer_id}/features` | Recupera as features de um cliente para edição. |
| `POST /predict/features` | Calcula o score a partir das features fornecidas. |
| `POST /predict/customer/{customer_id}` | Recupera o cliente na ABT e calcula o score. |

### Carregamento do modelo e health check

A inicialização do processo da API não depende da presença imediata do artefato.
Ao subir, o serviço cria uma tarefa em segundo plano para carregar o arquivo
indicado por `MODEL_PATH`. Se o arquivo estiver ausente, corrompido ou incompatível,
a exceção e o caminho são registrados no log do `credit-api`. A tarefa aguarda o
valor definido em `MODEL_LOAD_RETRY_SECONDS` e tenta novamente, sem exigir a
recriação do container.

Enquanto a carga não for concluída, `GET /health` funciona como uma verificação de
prontidão e responde HTTP `503`:

```json
{
  "detail": {
    "status": "unavailable",
    "model_loaded": false,
    "model_path": "/app/Model/artifacts/lightgbm_abt.pkl",
    "message": "O artefato do modelo ainda não foi carregado.",
    "last_error": "Modelo não encontrado: /app/Model/artifacts/lightgbm_abt.pkl"
  }
}
```

O mesmo código `503` protege `/model/features` e os endpoints de predição contra
uso antes da carga. Quando uma tentativa termina com sucesso, o objeto permanece
em memória para as requisições seguintes e `/health` passa automaticamente a
responder HTTP `200`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "/app/Model/artifacts/lightgbm_abt.pkl"
}
```

Esse desenho mantém a API observável durante a indisponibilidade, sem declarar o
serviço pronto antes que ele consiga realizar predições.

### Estrutura da requisição por features

O endpoint recebe um objeto `features` com todas as entradas listadas por `GET /model/features`:

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
  "risk_score": 0.55,
  "predicted_class": 1,
  "policy": {
    "recommendation": "manual_review",
    "reason": "Score na faixa intermediária; requer análise humana.",
    "policy_version": "demo-v1",
    "approve_max_score": 0.50,
    "manual_review_max_score": 0.60
  }
}
```

`source` informa se a pontuação veio do formulário ou do banco. Quando a consulta parte de um cliente armazenado, `customer_id` permite associar o resultado à origem.

## Jornadas do frontend

O Streamlit implementa três formas de demonstração:

### Preencher todos os dados

Renderiza as features agrupadas por contexto. Campos categóricos usam opções controladas, flags usam seleção binária e valores numéricos respeitam limites e passos definidos em `field_config.py`.

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

## Próximos passos

Além de calibração do score, autenticação e adoção de um *model registry*, dois eixos completam a proposta de arquitetura (itens iii e iv do escopo individual).

### iii. Monitoramento em produção

O objetivo é detectar **falhas, perda de performance e mudança de comportamento dos dados** antes que afetem a decisão de crédito. Como a base é **transversal (sem datas absolutas de originação)**, o monitoramento é definido por **lote de novas aplicações comparado ao baseline de treino** — e não por safra temporal, que exigiria coortes datadas inexistentes neste conjunto. Cada dimensão tem um **alerta** que aciona reavaliação ou re-treino:

- **estabilidade dos dados** — PSI do score e das principais features de cada novo lote contra a população de treino (não depende de rótulos nem de datas);
- **desempenho** — AUC/KS recalculados à medida que os desfechos (inadimplência) dos aprovados amadurecem, contra o baseline do teste;
- **decisão** — taxa de aprovação e inadimplência observada dos aprovados por lote;
- **calibração** — Brier / curva de calibração conforme os desfechos são observados;
- **fairness** — desempenho e taxa de negados por subgrupo.

### iv. Ações automatizadas a partir das previsões

As predições podem **acionar ações** de negócio, conectando ML, automação e agentes de IA:

- **roteamento automático** do pedido conforme a faixa da política (aprovação direta, fila de revisão humana, recusa justificada);
- **priorização da fila** de análise pelos casos de maior risco/valor;
- **agente de IA** que compõe um resumo explicável da decisão (drivers SHAP + política aplicada) para o analista;
- **gatilho de re-treino** aberto automaticamente quando um alerta de drift ou queda de performance dispara.

Essas ações permanecem **sob supervisão humana**: o modelo ordena risco e recomenda; a concessão final segue a política e a análise do analista.

## Componentes relacionados

- [Modelo](../Model/README.md)
- [PostgreSQL](../postgres/README.md)
- [Airflow](../airflow/README.md)
- [Arquitetura da plataforma](../README.md)
