import streamlit as st
import numpy as np
import joblib

model = joblib.load("app/models/lgbm_composite.pkl")

st.title("🔮 Predição de Risco de Defasagem")

st.write("Preencha os indicadores do aluno para estimar o risco.")

ida = st.slider("IDA", 0.0, 10.0, 5.0)
ieg = st.slider("IEG", 0.0, 10.0, 5.0)
ips = st.slider("IPS", 0.0, 10.0, 5.0)
iaa = st.slider("IAA", 0.0, 10.0, 5.0)
ipv = st.slider("IPV", 0.0, 10.0, 5.0)

if st.button("Calcular risco"):
    X = np.array([[ida, ieg, ips, iaa, ipv]])

    prob = model.predict_proba(X)[0][1]

    st.metric("Probabilidade de risco", f"{prob*100:.1f}%")

    if prob < 0.3:
        st.success("🟢 Baixo risco")
    elif prob < 0.6:
        st.warning("🟡 Médio risco")
    else:
        st.error("🔴 Alto risco")