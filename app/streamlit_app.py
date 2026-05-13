import streamlit as st

st.set_page_config(
    page_title="Passos Mágicos",
    page_icon="🎓",
    layout="wide"
)

# =========================
# HERO
# =========================

st.markdown("""
    <style>
    .hero {
        padding: 3rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .card {
        background-color: #111827;
        padding: 1.5rem;
        border-radius: 18px;
        border: 1px solid #374151;
        height: 100%;
    }

    .card h3 {
        color: white;
    }

    .card p {
        color: #D1D5DB;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🎓 Passos Mágicos</h1>
    <p>
        Plataforma analítica para monitoramento educacional,
        identificação de risco de defasagem e apoio à tomada de decisão.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# KPIs
# =========================

col1, col2, col3, col4 = st.columns(4)

col1.metric("👩‍🎓 Alunos Monitorados", "1.300+")
col2.metric("📈 Precisão do Modelo", "84%")
col3.metric("⚠️ Risco Detectado", "Tempo real")
col4.metric("🧠 Modelo Utilizado", "LightGBM")

st.divider()

# =========================
# SOBRE O PROJETO
# =========================

st.header("📚 Sobre o Projeto")

st.write("""
Este projeto foi desenvolvido para o Datathon da Passos Mágicos com o objetivo de:

- Monitorar indicadores educacionais;
- Identificar alunos em risco de defasagem;
- Analisar evolução acadêmica;
- Gerar insights para apoio pedagógico;
- Utilizar Machine Learning para prevenção de evasão e queda de desempenho.
""")

# =========================
# CARDS
# =========================

st.header("🚀 Funcionalidades")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <h3>📊 Dashboard Executivo</h3>
        <p>
            Visualização de indicadores educacionais,
            evolução temporal e métricas estratégicas.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/1_Dashboard.py",
        label="Acessar Dashboard",
        icon="📊"
    )

with c2:
    st.markdown("""
    <div class="card">
        <h3>👤 Análise Individual</h3>
        <p>
            Acompanhamento detalhado da evolução dos alunos
            ao longo dos anos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/2_Analise_Aluno.py",
        label="Acessar Análise",
        icon="👤"
    )

with c3:
    st.markdown("""
    <div class="card">
        <h3>🔮 Predição de Risco</h3>
        <p>
            Modelo preditivo baseado em Machine Learning
            para identificação precoce de risco.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.page_link(
        "pages/3_Predicao_Risco.py",
        label="Acessar Predição",
        icon="🔮"
    )

# =========================
# RODAPÉ
# =========================

st.caption(
    "Desenvolvido para o Datathon Passos Mágicos • FIAP Pós-Tech • Data Analytics"
)