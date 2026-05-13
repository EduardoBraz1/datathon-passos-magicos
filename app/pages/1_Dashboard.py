import streamlit as st
import pandas as pd
import plotly.express as px

st.page_link(
    "streamlit_app.py",
    label="⬅️ Voltar para Home",
    icon="🏠"
)

# título
st.title("📊 Dashboard Executivo")

# carregar base
df = pd.read_csv("data/processed/base_historico.csv")

# KPIs
col1, col2, col3 = st.columns(3)

col1.metric(
    "Total de alunos",
    df["Nome Anonimizado"].nunique()
)

col2.metric(
    "Média IDA",
    round(df["IDA"].mean(), 2)
)

col3.metric(
    "Média IEG",
    round(df["IEG"].mean(), 2)
)

# gráfico distribuição IDA
fig = px.histogram(
    df,
    x="IDA",
    title="Distribuição do IDA",
    nbins=20
)

st.plotly_chart(fig, use_container_width=True)

# gráfico evolução por ano
media_ano = df.groupby("Ano")["INDE"].mean().reset_index()

fig2 = px.line(
    media_ano,
    x="Ano",
    y="INDE",
    markers=True,
    title="Evolução média do INDE"
)

st.plotly_chart(fig2, use_container_width=True)