"""Interface Streamlit para a API de risco de crédito."""

import os
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

DATA_PLATFORM_DIR = Path(__file__).resolve().parents[3]
if str(DATA_PLATFORM_DIR) not in sys.path:
    # O Streamlit executa o arquivo como script e não inclui a raiz do projeto.
    sys.path.insert(0, str(DATA_PLATFORM_DIR))

from MLOps.app.frontend.field_config import FIELDS, GROUPS, FieldConfig


DEFAULT_API_URL = os.getenv("CREDIT_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 30

RECOMMENDATIONS = {
    "approve": ("Aprovação recomendada", "success", "✅"),
    "manual_review": ("Revisão manual recomendada", "warning", "⚠️"),
    "reject": ("Reprovação recomendada", "error", "⛔"),
}


def api_request(method: str, url: str, **kwargs: Any) -> Any:
    """Executa uma chamada e converte erros HTTP em mensagens legíveis."""
    try:
        response = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError as error:
        raise RuntimeError("Não foi possível conectar à API. Verifique se a FastAPI está em execução.") from error
    except requests.Timeout as error:
        raise RuntimeError("A API demorou mais que o esperado para responder.") from error
    except requests.HTTPError as error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"A API retornou HTTP {response.status_code}: {detail}") from error


def render_input(field: FieldConfig) -> Any:
    """Renderiza o componente adequado e devolve um valor serializável em JSON."""
    key = f"feature_{field.name}"
    if field.kind == "category":
        index = field.options.index(str(field.default))
        return st.selectbox(field.label, field.options, index=index, key=key, help=field.help or None)
    if field.kind == "boolean":
        value = st.selectbox(field.label, ("Não", "Sim"), index=int(field.default), key=key)
        return int(value == "Sim")

    arguments: dict[str, Any] = {
        "label": field.label,
        "value": int(field.default) if field.kind == "integer" else float(field.default),
        "step": int(field.step) if field.kind == "integer" else float(field.step),
        "key": key,
        "help": field.help or None,
    }
    if field.minimum is not None:
        arguments["min_value"] = int(field.minimum) if field.kind == "integer" else float(field.minimum)
    if field.maximum is not None:
        arguments["max_value"] = int(field.maximum) if field.kind == "integer" else float(field.maximum)
    value = st.number_input(**arguments)
    return int(value) if field.kind == "integer" else float(value)


def show_result(result: dict[str, Any]) -> None:
    """Apresenta score, classificação do modelo e política de crédito."""
    policy = result["policy"]
    recommendation = policy["recommendation"]
    title, message_type, icon = RECOMMENDATIONS.get(
        recommendation, (recommendation, "info", "ℹ️")
    )

    st.subheader(f"{icon} {title}")
    getattr(st, message_type)(policy["reason"])

    score = float(result["risk_score"])
    col_score, col_class, col_source = st.columns(3)
    col_score.metric("Score de risco", f"{score:.2%}")
    col_class.metric("Classe prevista", "Inadimplente" if result["predicted_class"] else "Adimplente")
    col_source.metric("Origem", "Banco de dados" if result["source"] == "database" else "Formulário")
    st.progress(score, text="Posição do cliente na escala de risco do modelo")

    st.caption(
        f"Limiar do modelo: {result['model_decision_threshold']:.2f} · "
        f"Política: {policy['policy_version']} · "
        f"Aprovar abaixo de {policy['approve_max_score']:.2f} · "
        f"Reprovar a partir de {policy['manual_review_max_score']:.2f}"
    )
    st.info(
        "Este score é uma pontuação para ordenação de risco e não uma probabilidade "
        "calibrada de inadimplência. A recomendação é demonstrativa e requer validação humana."
    )
    with st.expander("Resposta completa da API"):
        st.json(result)


st.set_page_config(page_title="Análise de Crédito", page_icon="💳", layout="wide")
st.title("Análise de risco de crédito")
st.caption("Simulador para apoio ao analista de crédito · Projeto acadêmico")

with st.sidebar:
    st.header("Configuração")
    api_url = st.text_input("URL da FastAPI", value=DEFAULT_API_URL).rstrip("/")
    if st.button("Verificar conexão", use_container_width=True):
        try:
            health = api_request("GET", f"{api_url}/health")
            if health.get("status") == "ok" and health.get("model_loaded"):
                st.success("API conectada e modelo carregado.")
            else:
                st.warning("A API respondeu, mas o modelo não está disponível.")
        except RuntimeError as error:
            st.error(str(error))
    st.divider()
    st.caption("A interface envia os dados à API. O modelo permanece isolado no backend.")

tab_features, tab_customer = st.tabs(("Preencher todos os dados", "Consultar cliente do banco"))

with tab_features:
    st.write("Preencha as informações esperadas pelo endpoint `POST /predict/features`.")
    with st.form("credit_features_form"):
        features: dict[str, Any] = {}
        for group in GROUPS:
            st.subheader(group)
            group_fields = [field for field in FIELDS if field.group == group]
            columns = st.columns(3)
            for index, field in enumerate(group_fields):
                with columns[index % 3]:
                    features[field.name] = render_input(field)
        submitted = st.form_submit_button("Analisar crédito", type="primary", use_container_width=True)

    if submitted:
        payload = {"features": features}
        with st.expander("JSON enviado à API"):
            st.json(payload)
        try:
            with st.spinner("Calculando o score de risco..."):
                result = api_request("POST", f"{api_url}/predict/features", json=payload)
            show_result(result)
        except RuntimeError as error:
            st.error(str(error))

with tab_customer:
    st.write("Informe um identificador existente para usar o endpoint `POST /predict/customer/{customer_id}`.")
    with st.form("customer_id_form"):
        customer_id = st.number_input("Código do cliente", min_value=1, value=100002, step=1)
        customer_submitted = st.form_submit_button("Consultar e analisar", type="primary", use_container_width=True)

    if customer_submitted:
        try:
            with st.spinner("Consultando as fontes e calculando o risco..."):
                result = api_request("POST", f"{api_url}/predict/customer/{int(customer_id)}")
            show_result(result)
        except RuntimeError as error:
            st.error(str(error))
