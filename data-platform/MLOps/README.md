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

### Onde ficam o compose e a orquestração

A especificação da etapa individual lista `docker-compose` e `pipeline_orchestration.py` ao lado de `MLOps`. Nesta solução eles ficam no **nível da plataforma**, porque orquestram todos os serviços — não apenas o MLOps:

| Artefato do escopo | Local neste repositório |
|---|---|
| `docker-compose` | [`../docker-compose.yml`](../docker-compose.yml) — sobe Postgres, Airflow, API e frontend |
| `pipeline_orchestration.py` | [`../airflow/dags/pipeline_orchestration.py`](../airflow/dags/pipeline_orchestration.py) — DAG bruta → clean → ABT → treino |
| `predict.py` | [`../Model/predict.py`](../Model/predict.py) — serviço de predição local (CLI) |

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

Além de calibração do score, autenticação e adoção de um *model registry*, dois eixos completam a proposta de arquitetura, atendendo aos itens iii e iv do escopo individual. Os dois são especificação de próxima fase, não implementação, mas foram desenhados sobre os componentes e os números que este projeto já produz.

### iii. Monitoramento dos dados e do modelo em produção

Um serviço convencional quebra fazendo barulho: erro HTTP, task vermelha no Airflow, exceção no log. Um modelo de machine learning tem um segundo modo de falha, mais perigoso, porque é silencioso: ele continua respondendo scores com aparência normal mesmo quando a população mudou ou quando a fonte de uma feature degradou. Ninguém percebe até o prejuízo aparecer na carteira, meses depois. O plano de monitoramento existe para encurtar esse tempo.

#### Possíveis falhas

É a camada mais simples e a primeira a cobrir, porque a causa mais comum de score errado em produção não é estatística, é operacional: pipeline que falha no meio, artefato que não carrega, fonte que muda de formato.

Parte dos sensores já existe nesta plataforma: o `/health` da API distingue "processo no ar" de "modelo carregado" (respondendo `503` enquanto o artefato não carrega) e o Airflow registra o status de cada task da DAG. O que a proposta adiciona é a observação sistemática sobre esses sensores, que hoje não existe: medir a taxa de `503` e a latência de predição ao longo do tempo, alertar quando uma task falha e verificar a integridade dos dados de entrada (taxa de nulos por feature e aparecimento de categorias desconhecidas, comparadas com o treino) antes de pontuar.

#### Mudanças de comportamento dos dados

O modelo foi treinado com uma fotografia da população. Se o perfil de quem pede crédito mudar (uma crise econômica, uma campanha que atrai outro público, etc) o modelo segue respondendo com confiança sobre uma população que nunca viu. Para detectar isso sem depender de rótulo, podemos comparar a distribuição dos dados novos com a do treino usando o PSI (*Population Stability Index*), calculado por lote de aplicações, usando os decis do score de treino como bins fixos (a mesma partição da tabela de decis do `evaluation.ipynb`).

Os limiares adotados são convenção de mercado, não derivados desta base (calibrar isso exigiria um histórico que não temos no momento): 
- abaixo de 0,10, população estável; 
- entre 0,10 e 0,25, atenção; 
- acima de 0,25, mudança relevante e gatilho de investigação.

O PSI é calculado para o score e para as features de maior impacto tendo em primeiro lugar a `ext_source_mean`, porque a análise de permutação mostrou que ela sustenta o modelo quase sozinha, cerca de treze vezes a segunda colocada. Uma quebra na fonte desse score externo é o maior risco isolado deste modelo, e por isso seria o primeiro alerta a ser implementado.

Foi optado monitorar por lote, e não por safra temporal, porque a base Home Credit é transversal: não há data de origem. Num cenário real com datas, a mesma lógica passa a rodar por safra sem mudar de estrutura, muda só a chave de agrupamento. Ou seja, deixaria de agrupar pela semana em que foi pontuado, e passaria a agrupar pelo mês em que o contrato foi originado.

#### Perda de performance

A variável resposta chega tarde. Só sabemos se um cliente aprovado é bom ou mau pagador meses depois da concessão, não existe "AUC de ontem". Por isso a detecção acontece em dois tempos.

No curto prazo, sem rótulo, dá para usar indicadores indiretos: se a taxa de aprovação sair da casa dos ~67% do baseline (proponho ±5 p.p.), ou se o volume da faixa de revisão humana subir muito além dos 12,3%, algo mudou — na população ou no pipeline — e dá para descobrir logo, não após meses.

No longo prazo, quando a base amadurece, compara-se AUC/KS e o default real dos aprovados com o baseline; uma queda superior a 10% é gatilho de retreino.

Essa comparação de longo prazo tem uma armadilha que é assumida desde já: o viés de seleção. Só consigo observar o desfecho de quem foi aprovado. Um pedido negado nunca vira bom ou mau pagador, porque o empréstimo não aconteceu. Qualquer AUC recalculado em produção estará restrito à população com score abaixo do corte e não pode ser comparado diretamente com o que foi medido na distribuição inteira. Ignorar isso leva a concluir que o modelo piorou quando na verdade mudou a régua. As saídas possíveis, em ordem de custo: 
1) recalcular a referência na mesma fatia da população. O holdout guarda score e rótulo de todos os clientes — inclusive dos que seriam negados, porque na base histórica o desfecho de todo mundo é conhecido. O recálculo funciona assim: filtra-se o holdout mantendo apenas os clientes com score abaixo do corte (os que teriam sido aprovados) e recalcula-se o AUC somente nessa fatia. Esse número é naturalmente menor que o da distribuição inteira, porque os casos extremos e fáceis de ordenar (score muito alto ou muito baixo) ficam de fora — e é ele a régua justa para comparar com o AUC de produção, que só enxerga essa mesma fatia. É a saída mais barata das três: o holdout já existe salvo, com scores e rótulos, então o recálculo é só um filtro e uma nova medição;
2) aprovar deliberadamente uma amostra pequena na zona de recusa para gerar rótulo sem viés (grupo de controle). Tem custo real, porém, com valor estatístico alto; 
3) *reject inference*, que é a prática de mercado, mas depende de premissas fortes. Nesse caso, é inferido o que os negados teriam feito.

#### Como isso rodaria na plataforma atual

Nenhum componente novo é necessário. A API já registra cada predição em JSON no stdout do container; o primeiro passo é persistir esse log numa tabela do PostgreSQL. Uma DAG de monitoramento no Airflow passa a rodar por lote, calcula PSI, mix de decisão e taxas, compara com o baseline e grava o resultado numa tabela de métricas versionada (`model_version` e `policy_version`). O acompanhamento vira uma consulta SQL (ou um painel de BI sobre essas tabelas). Quando um alerta dispara, a investigação é humana; se a conclusão for retreinar, a DAG `pipeline_orchestration` é executada novamente.

### iv. Ações automatizadas a partir das previsões

Neste projeto, a primeira camada de automação já está construída, que é a própria política de crédito:

| Faixa de score | Recomendação | Ação automatizada | Volume (holdout) |
|---|---|---|---|
| < 0,50 | `approve` | esteira de aprovação e comunicação ao cliente, sem intervenção humana | ~67% |
| 0,50 a 0,60 | `manual_review` | entra na fila de análise humana, com dossiê preparado por agente | ~12,3% |
| ≥ 0,60 | `reject` | recusa automática com justificativa; contestação escala para humano | ~20,5% |

Cerca de 88% dos pedidos são decididos sem nenhum humano. O ganho de automação já aconteceu nas pontas; o custo operacional que resta está concentrado nos 12,3% do meio e é aí que o agente de IA passa a fazer sentido.

#### O agente de apoio à revisão humana

O analista que recebe um caso de revisão recebe hoje, na prática, um JSON com 42 features e um score. A proposta do agente é a seguinte: um modelo de linguagem recebe a resposta que a API já produz (score, recomendação, limiares, versão da política) mais os dados cadastrais autorizados, e escreve o dossiê em linguagem natural com informações relevantes para a tomada de decisão do analista:
- em que faixa o cliente caiu;
- como ele se compara com a população;
- o que a política determina. 

Ou seja, um tradutor de técnico para humano. O agente não produz nenhum número que não tenha recebido pronto.

Em crédito um número alucinado por um LLM que influencie uma negação indevida pode virar passivo jurídico.
Um dossiê mal escrito que o analista ignora custa quase nada. Já um número inventado pelo LLM que induza o analista a negar crédito indevidamente pode ter impactos jurídicos. Por isso o modelo ordena (score), a política enquadra (faixa), o analista decide (concessão), o agente escreve (texto). Cada camada faz uma coisa e a responsabilidade da decisão fica no humano.

Além do dossiê, dois gatilhos determinísticos completam a automação da fila: 
1) a ordenação por risco × exposição (`risk_score` × `amt_credit`), para que o analista olhe primeiro onde há mais dinheiro em jogo, e não quem chegou primeiro; 
2) e a contraproposta automática para recusados próximos do corte (limite menor, prazo maior), convertendo parte das negações em receita com risco controlado.

Hoje a API não devolve os drivers individuais de cada score. Incluir a explicação local (SHAP) na resposta daria ao agente o "porquê" de cada caso, além do "quanto". O dossiê do agente deixa de ser descritivo ("caiu na faixa de revisão") e vira explicativo ("caiu na revisão principalmente por X e Y").

#### Governança do conteúdo gerado

Este é o ponto mais sensível da proposta, e nasce da análise de fairness do próprio projeto. O modelo usa `code_gender`, e a taxa de negação é maior para homens, acompanhando a inadimplência real de cada grupo. 
Para o score, isso foi tratado como decisão de modelagem. Para o texto gerado, o tratamento precisa ser mais duro: gênero não pode aparecer como justificativa em nenhum dossiê ou comunicação, porque estatística interna auditável é uma coisa e justificativa declarada é discriminação.

A implementação proposta é uma marcação de citabilidade por feature, aplicada no código antes da chamada ao modelo de linguagem: os campos marcados como não citáveis (caso do `code_gender`) são removidos dos dados antes do envio. A alternativa — enviar tudo e instruir no prompt "não mencione o gênero" — é frágil, porque instrução textual é um pedido que o modelo pode eventualmente descumprir. Já o campo removido nunca chega ao modelo, e ele não tem como citar o que não recebeu.

Completando a governança, cada dossiê gerado registra a versão do modelo, da política e do prompt que o produziram. Sem esse rastro, um relatório contestado meses depois não teria como ser reconstituído; com ele, é possível saber exatamente que combinação de score, regra e instrução gerou cada texto.

#### Comportamento em falha

O agente depende de um provedor externo de LLM, e essa dependência não pode ficar no caminho crítico da predição. A API publica a solicitação de dossiê de forma assíncrona (uma fila simples resolve, ou, numa versão ainda mais enxuta, o próprio log de predições persistido serve de fonte para processamento posterior) e devolve o score sem esperar. Se o LLM estiver indisponível, a predição continua funcionando e o dossiê fica pendente sendo gerado quando o LLM voltar.

#### Conexão com o monitoramento

Os itens iii e iv se fecham num ciclo. Um alerta de PSI acima de 0,25 ou uma queda confirmada de performance pode disparar uma nova execução da DAG de treino. Já a publicação do modelo novo permanece decisão humana, com comparação de métricas antes da troca. É a mesma divisão defendida ao longo de toda a proposta: a automação prepara (retreina, calcula, organiza evidências) e a pessoa decide (investiga o alerta, publica o modelo, concede o crédito).

## Componentes relacionados

- [Modelo](../Model/README.md)
- [PostgreSQL](../postgres/README.md)
- [Airflow](../airflow/README.md)
- [Arquitetura da plataforma](../README.md)
