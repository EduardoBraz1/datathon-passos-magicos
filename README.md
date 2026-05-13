# 🪄 Datathon FIAP: ONG Passos Mágicos - Diagnóstico e Predição de Risco

## 📌 O Projeto
Este repositório contém a solução completa (Data Analytics & Machine Learning) desenvolvida para o Datathon da FIAP, focado na ONG Passos Mágicos. 

Nosso objetivo foi além de analisar o passado; nós criamos um sistema capaz de entender o que leva um aluno ao "Ponto de Virada" (IPV) e um **Modelo Preditivo de Risco** para alertar a ONG *antes* que um aluno caia de rendimento ou aumente sua defasagem escolar.

## 📊 Parte 1: Diagnóstico e Storytelling (Power BI & EDA)
Através da Análise Exploratória e do nosso Painel Executivo, diagnosticamos 4 alavancas cruciais da operação:
1. **O Gargalo da Fase 3:** Queda brusca de desempenho (IDA) no 7º/8º ano, a fase de maior risco.
2. **O Custo Emocional da Defasagem:** Provamos a correlação direta entre o atraso escolar e a deterioração da saúde psicossocial (IPS).
3. **O Viés de Otimismo (+1.55 pts):** Descobrimos que os alunos superestimam suas notas (IAA) em relação à realidade (IDA).
4. **A Batalha dos Profissionais:** A avaliação do professor (IPP - 0.37) tem peso estatisticamente maior no sucesso do aluno do que o contexto social (IPS - 0.02).

## 🤖 Parte 2: Predição de Risco (Machine Learning)
Desenvolvemos um pipeline robusto de Machine Learning para prever o risco futuro dos alunos em 2025:
* **Modelos Utilizados:** Ensembles avançados com LightGBM e XGBoost, além de Redes Neurais (MLP), otimizados via Optuna.
* **Explicabilidade (SHAP):** O modelo não é uma "caixa preta". Utilizamos SHAP values para explicar à ONG *por que* um aluno foi classificado como alto risco, indicando os fatores exatos para intervenção pedagógica.
* **Múltiplos Alvos:** Predição focada em risco de queda no INDE, piora na Defasagem e queda do conceito da Pedra.

## 📁 Estrutura do Repositório
* `app/`: Aplicação Web desenvolvida em Streamlit (Dashboard, Análise Individual e Predições).
* `data/`: Bases de dados brutas e processadas.
* `docs/`: Arquivos do Power BI (`.pbix`) e documentações detalhadas do modelo (`MODELO_RISCO.md`).
* `notebooks/`: Análises exploratórias e laboratório de dados (EDA).
* `src/risk_model/`: Código-fonte do pipeline de Machine Learning (treinamento, avaliação, predição e features).
* `scripts/`: Scripts automatizados para rodar o pipeline de dados e treinamento.

## 🚀 Como acessar o projeto (Live)
* **Web App (Streamlit):** [https://datathon-paapps-magicos-nk5svdeyhty9btfdrsovfc.streamlit.app/](https://datathon-paapps-magicos-nk5svdeyhty9btfdrsovfc.streamlit.app/)
* **Dashboard Executivo (Power BI):** [https://app.powerbi.com/view?r=eyJrIjoiOTI5ZmJmYjMtYTYxYy00MDE2LThlZmMtMTczOWMyZjM2MWRmIiwidCI6ImM5NjhjZDA5LTZlODgtNDVjZi1hMzliLWQwYmExMjdjZGNmYiJ9](https://app.powerbi.com/view?r=eyJrIjoiOTI5ZmJmYjMtYTYxYy00MDE2LThlZmMtMTczOWMyZjM2MWRmIiwidCI6ImM5NjhjZDA5LTZlODgtNDVjZi1hMzliLWQwYmExMjdjZGNmYiJ9)

## 💻 Como executar localmente
1. Clone este repositório:
   ```bash
   git clone [https://github.com/EduardoBraz1/datathon-passos-magicos.git](https://github.com/EduardoBraz1/datathon-passos-magicos.git)
   cd datathon-passos-magicos
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute a aplicação:
   ```bash
   streamlit run app/streamlit_app.py
   ```
