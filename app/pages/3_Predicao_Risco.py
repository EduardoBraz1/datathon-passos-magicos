import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# CARREGAR MODELO
# =========================

model_list = joblib.load("app/models/lgbm_composite.pkl")
model = model_list[0]

prep = joblib.load("app/models/preprocessor.pkl")

feature_cols = prep["feature_cols"]

# =========================
# INTERFACE
# =========================

st.title("🔮 Predição de Risco de Defasagem")

st.write("""
Preencha os principais indicadores do aluno.
""")

# INPUTS
idade = st.slider("Idade", 7, 25, 12)

fase = st.slider("Fase", 0, 10, 2)

ano_ingresso = st.slider("Ano ingresso", 2015, 2025, 2021)

inde = st.slider("INDE", 0.0, 10.0, 7.0)

iaa = st.slider("IAA", 0.0, 10.0, 7.0)

ieg = st.slider("IEG", 0.0, 10.0, 7.0)

ips = st.slider("IPS", 0.0, 10.0, 7.0)

ida = st.slider("IDA", 0.0, 10.0, 7.0)

ipv = st.slider("IPV", 0.0, 10.0, 7.0)

ian = st.slider("IAN", 0.0, 10.0, 7.0)

# =========================
# PREDIÇÃO
# =========================

if st.button("Calcular risco"):

    # cria dataframe vazio
    X = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols
    )

    # preencher principais features
    X["Idade"] = idade
    X["Fase"] = fase
    X["Ano ingresso"] = ano_ingresso

    X["INDE"] = inde
    X["IAA"] = iaa
    X["IEG"] = ieg
    X["IPS"] = ips
    X["IDA"] = ida
    X["IPV"] = ipv
    X["IAN"] = ian

    # alguns defaults úteis
    X["had_prev_year"] = 1
    X["years_in_program"] = 2

    # gênero
    X["Gênero_Feminino"] = 1
    X["Gênero_Masculino"] = 0

    # escola
    X["Instituição de ensino_Publica"] = 1

    # predição
    prob = model.predict_proba(X)[0][1]

    # =========================
    # RESULTADO
    # =========================

    st.subheader("Resultado")

    st.metric(
        "Probabilidade de risco",
        f"{prob*100:.1f}%"
    )

    # semáforo
    if prob < 0.3:
        st.success("🟢 Baixo risco")

    elif prob < 0.6:
        st.warning("🟡 Médio risco")

    else:
        st.error("🔴 Alto risco")

    # insights
    st.subheader("Principais fatores")

    fatores = []

    if ida < 5:
        fatores.append("Baixo desempenho acadêmico")

    if ieg < 5:
        fatores.append("Baixo engajamento")

    if ips < 5:
        fatores.append("Aspectos psicossociais críticos")

    if iaa < 5:
        fatores.append("Baixa autoavaliação")

    if ipv < 5:
        fatores.append("Baixo ponto de virada")

    if len(fatores) == 0:
        st.write("Nenhum fator crítico identificado.")

    else:
        for fator in fatores:
            st.write(f"• {fator}")