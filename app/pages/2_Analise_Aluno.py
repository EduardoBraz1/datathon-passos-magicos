import streamlit as st
import pandas as pd
import plotly.express as px

st.page_link(
    "streamlit_app.py",
    label="⬅️ Voltar para Home",
    icon="🏠"
)

st.title("👤 Análise Individual do Aluno")

# carregar dados
df = pd.read_csv("data/processed/base_historico.csv")

# selecionar aluno
aluno = st.selectbox(
    "Selecione o aluno",
    sorted(df["Nome Anonimizado"].unique())
)

# filtrar
dados_aluno = df[df["Nome Anonimizado"] == aluno]

# mostrar tabela
st.subheader("📋 Dados do aluno")

st.dataframe(dados_aluno)

# evolução temporal
st.subheader("📈 Evolução do INDE")

fig = px.line(
    dados_aluno,
    x="Ano",
    y="INDE",
    markers=True,
    title=f"Evolução do INDE - {aluno}"
)

st.plotly_chart(fig, use_container_width=True)

# radar simplificado
st.subheader("🎯 Indicadores médios")

indicadores = dados_aluno[
    ["IDA", "IEG", "IPS", "IAA", "IPV"]
].mean()

st.bar_chart(indicadores)