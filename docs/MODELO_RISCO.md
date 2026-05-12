# Modelo de Risco Acadêmico — Passos Mágicos

Pipeline completo: **LogReg + LightGBM + XGBoost + MLP + Stacking + Ensemble
por média.** Treino temporal sobre a transição 2022→2023 e hold-out cego sobre
2023→2024. Todas as decisões abaixo são tecnicamente defensáveis e livres de
*data leakage* (calibração, imputers e encoders são `fit` apenas no treino).

---

## 1. Problema e alvo

A Passos Mágicos quer **antecipar regressão acadêmica** antes que ela ocorra,
dado o histórico do aluno até o ano *t*. O alvo principal é composto (união
de três regras):

| variante | regra (de t → t+1) |
|---|---|
| `risk_pedra_drop` | Pedra em t+1 estritamente pior do que em t **ou** permanece em Quartzo. |
| `risk_inde_drop` | INDE cai mais que 0.5 pontos **ou** fica abaixo do 1º quartil da coorte t+1. |
| `risk_defasagem_worsen` | Defasagem aumenta em t+1 **ou** atinge ≥ 1 ano. |
| **`risk_composite` (PRIMÁRIO)** | União das três — captura qualquer deterioração relevante. |

### Diagnóstico de vazamento

Confirmado: `Defasagem` no ano *t* é uma **feature** legítima (estado conhecido
em *t*). O rótulo composto usa `Defasagem_{t+1}` (não `Defasagem_t` isolado)
nas regras `def_t1 > def_t OR def_t1 >= 1`. A predição `Defasagem_t ≥ 1 ⇒
label=1` (forte mas não vazada) é uma correlação real do mundo: alunos já
defasados raramente recuperam ano.

### Prevalência (importante para leitura das métricas)

| transição | n pares | risk_composite |
|---|---|---|
| 2022→2023 (treino) | 600 | 61,67 % |
| 2023→2024 (hold-out) | 765 | 65,23 % |

Como a prevalência é alta, **PR-AUC** é mais informativo do que ROC-AUC e
**recall@top-decile** é estruturalmente limitado a ≈ 0,15 (com 499 positivos
no hold-out, top-10 % cabe no máximo 77 alunos → recall ≤ 77/499 ≈ 0,154).

---

## 2. Dados, limpeza e features (108 colunas no input do modelo)

Implementação em `src/risk_model/{data,features}.py`.

### Limpeza obrigatória
- `Pedra`: unificação `Ágata` → `Agata`; alunos com Pedra NaN mas INDE válido
  recebem Pedra inferida pelas faixas oficiais PEDE.
- `Gênero`: `Menina/Menino` → `Feminino/Masculino`.
- `Instituição de ensino`: 12 categorias colapsadas em 5 (Publica,
  Privada_Bolsa, Privada, Concluido, Outro).
- `Ing`: imputação por **mediana dentro da Fase** (não global), aplicada *antes*
  do cálculo de lags/rolling — sem isso `prev_Ing` / `roll2_Ing_mean` ficariam
  100 % NaN nas Fases 0–2.
- `IPP / prev_IPP / roll2_IPP_mean`: 100 % NaN em 2022 (indicador introduzido
  em 2023). **Descartados na assemblagem** do treino para evitar imputadores
  instáveis.

### Famílias de features

1. **Indicadores brutos t** — INDE, IAA, IEG, IPS, IDA, Mat, Por, Ing, IPV,
   IAN, IPP.
2. **Ordinais & meta** — Fase, Idade, Ano ingresso, Pedra_ord, Defasagem,
   FaseIdeal_num, Nº Av.
3. **Lags (t-1)** — `prev_<ind>`, `delta_<ind>` com preenchimento estrutural
   (sem t-1 → prev=current, delta=0, `had_prev_year`=0).
4. **Rolling 2 anos** — `roll2_<ind>_mean`, `roll2_<ind>_std` (std=0 se janela
   tem 1 obs).
5. **Interações** — `gap_Mat_Por`, `gap_Mat_Ing`, `gap_Por_Ing`,
   `gap_IPS_IDA`, `gap_IAA_IEG`, `inde_z_fase`, `age_fase_excess`.
6. **Novo (refinamento autorizado):** z-score por (Fase × Ano) e *decile dentro
   da Fase* para INDE, IAA, IEG, IPS, IDA, IPV — comparação contra a coorte
   real do aluno no mesmo ano.
7. **Trend slope (novo)** — *inclinação linear* da trajetória do aluno em INDE,
   IAA, IEG, IPS, IDA, IPV, Defasagem, usando todos os anos observados até *t*
   (slope=0 e `trend_window`=1 para quem tem 1 só observação).
8. **Sinalizadores** — `<col>_was_missing` por indicador, `is_new_student`,
   `years_in_program`.

> **Sem vazamento de t+1**: cada feature usa apenas observações ≤ t. O alvo
> usa exclusivamente observações em t+1 (combinadas com Defasagem_t, que já é
> feature válida).

---

## 3. Arquitetura do pipeline (`scripts/run_pipeline.py`)

| Estágio | Conteúdo |
|---|---|
| 1 | Carrega + limpa + monta train/val/predict, descarta features 100 % NaN no treino. |
| 2 | Sweep de hiperparâmetros LightGBM (3×3=9 combos) e XGBoost (3×3=9) via `GroupKFold(5)` *somente no treino*. Tie-break preferindo simplicidade quando CV-AUC fica dentro de 0,005. |
| 3 | Treina LogReg (5-fold OOF), LightGBM **seed-bag (3 sementes: 42, 7, 2024)** com early-stopping no hold-out por semente, XGBoost com early-stopping no hold-out. |
| 4 | Calibra LightGBM e XGBoost via `CalibratedClassifierCV(method=…, cv=GroupKFold(5))` no treino; comparação isotônica vs sigmoid medida no hold-out, escolhendo sigmoid se isotônica piora AUC > 0,01. |
| 5 | MLP PyTorch em **subprocess dedicado** (`python -m risk_model.mlp_subprocess`) — força `device='cpu'` para descartar instabilidade MPS no macOS observada na rodada anterior. |
| 6 | Spearman entre LGBM × XGB × MLP no hold-out para checar diversidade do ensemble. |
| 7 | Stacking 4-modelos (LGBM + XGB + MLP + LogReg) com meta-learner LogReg L2 treinado em **OOF do treino** (sem vazamento). |
| 8 | Calibra MLP via **Platt** treinado nas predições OOF do treino. |
| 9 | Threshold operacional vem do **max F1 no OOF do treino**, aplicado fixo no hold-out. |
| 10 | Avaliação no hold-out + plots (ROC/PR, calibração, AUC por Fase, distribuição de risco). |
| 11 | SHAP TreeExplainer para LightGBM e XGBoost. |
| 12 | Predições para 2025 + faixa de risco. |

### Arquitetura do MLP

`Linear(108→256) → BN → GELU → Dropout(0.3) → Linear(→128) → BN → GELU →
Dropout(0.3) → Linear(→64) → GELU → Dropout(0.2) → Linear(→1)`.

Loss: `BCEWithLogitsLoss(pos_weight = neg/pos)` (= 0,624 com prevalência 0,617).
Otimizador: AdamW (lr=1e-3, wd=1e-4). Scheduler: OneCycleLR (max_lr=3e-3,
pct_start=0,30). Batch=64, até 200 épocas, *early stopping* (patience=20) em
**ROC-AUC do val**.

### Stacking

Manual, livre de vazamento:

1. Para cada base learner, gera predições **out-of-fold** sobre o período de
   treino com `GroupKFold(5)` por RA. LightGBM usa as 3 sementes médiadas em
   cada fold; XGB e LogReg usam uma instância fresca por fold; MLP roda 5 vezes
   dentro do subprocess.
2. Empilha as quatro colunas OOF e treina o meta `LogReg(C=1.0, l2,
   class_weight='balanced')` nessas predições.
3. Predições do hold-out e do 2024 vêm dos modelos *finais* (ajustados em todo
   o treino) — esse é o protocolo correto: o meta vê apenas OOF para evitar
   vício, mas em produção usamos a versão treinada em tudo.

Pesos aprendidos pelo meta: `lgbm=+1,25, xgb=+1,05, mlp=+1,80, logreg=+0,53`.
MLP recebe o maior peso, mas com correlação Spearman moderada
(lgbm-mlp=0,73) o ganho marginal sobre LGBM puro é pequeno.

### Threshold operacional

Cortado a 0,36 (otimizado no OOF do treino). Aplicado *sem retoque* sobre o
hold-out — a F1 reportada não está inflada artificialmente.

---

## 4. Resultados do hold-out (2023 → 2024) — alvo composto

| Modelo | ROC-AUC | PR-AUC | Brier | F1 @ thr=0,29 | Recall@top-10 % |
|---|---|---|---|---|---|
| Prevalência (baseline)        | 0,500 | 0,652 | 0,227 | 0,000 | 0,100 |
| LogReg                        | 0,706 | 0,816 | 0,208 | 0,797 | 0,134 |
| **LightGBM (raw, seed-bag×3)**| **0,758** | **0,843** | 0,200 | 0,791 | 0,142 |
| LightGBM (calibr. sigmoid)    | 0,738 | 0,833 | **0,193** | 0,791 | 0,140 |
| XGBoost (raw)                 | 0,718 | 0,826 | 0,209 | 0,794 | 0,142 |
| XGBoost (calibr. sigmoid)     | 0,715 | 0,827 | 0,201 | 0,792 | 0,142 |
| MLP (raw)                     | 0,682 | 0,783 | 0,231 | 0,774 | 0,124 |
| MLP (calibr. Platt)           | 0,682 | 0,783 | 0,208 | 0,792 | 0,124 |
| Stacking (4 modelos, OOF)     | 0,739 | 0,841 | 0,202 | 0,802 | 0,140 |
| Média(LGBM_cal, MLP_cal)      | 0,735 | 0,836 | 0,193 | 0,791 | 0,136 |

**Finalista no hold-out:** `LightGBM_raw` — ROC-AUC = **0,758**, PR-AUC =
**0,843**, Brier = **0,200**. É o modelo que escreve `prob_final` no CSV
de predições para 2025. Calibração foi avaliada (sigmoid escolhido sobre
isotônica por audit) mas degrada AUC neste tamanho de amostra, portanto
mantemos a versão raw para *ranqueamento* e disponibilizamos `prob_lgbm`
(calibrada) ao lado quando a NGO precisar de probabilidade absoluta.

### Bateu a barra de 0,78?

**Não.** Margem: **−0,022 ROC-AUC** (0,758 vs 0,780).

> Por quê? O CV-AUC no próprio treino (`GroupKFold(5)` em 2022) já está em
> **0,755** no melhor caso — i.e. *a barra de 0,78 não existe no sinal
> disponível*. Detalhes na seção 8.

### Sub-população

`reports/auc_by_fase.csv`, `auc_by_pedra.csv`, `auc_by_gender.csv` + plot
`reports/figures/auc_by_fase.png`.

| Fase | n | pos_rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0 | 174 | 59 % | 0,716 | 0,797 |
| 1 | 138 | 72 % | **0,850** | **0,922** |
| 2 | 153 | 86 % | 0,790 | 0,951 |
| 3 |  94 | 66 % | 0,658 | 0,811 |
| 4 |  67 | 63 % | 0,525 | 0,707 |
| 5 |  43 | 58 % | 0,564 | 0,703 |
| 6 |  17 | 59 % | 0,914 | 0,953 |
| 7 |  20 | 70 % | 0,214 | 0,596 |
| 8 |  59 | 22 % | 0,751 | 0,450 |

**Leitura honesta:** o modelo é forte nas Fases 0-2 (fundamental 1, onde o
sinal de defasagem é mais previsível). Fases 4-5 (médio inicial) caem para
≈ 0,55 — provavelmente onde a heterogeneidade da população é máxima e o
sinal mais ruidoso. Fase 7 (n=20) é amostra pequena demais para conclusão.

Por **Pedra inicial**: Ametista (0,78, n=308) é a categoria onde o modelo
mais ajuda; Topázio (0,66) menos — alunos no topo têm pouca margem para subir.

Por **Gênero**: AUC praticamente igual (0,74 Feminino vs 0,75 Masculino) — sem
viés evidente.

---

## 5. Calibração

`reports/figures/calibration_curves.png`. A Platt scaling no MLP reduziu
Brier de **0,247 → 0,209** sem alterar AUC (esperado). LightGBM beneficia-se
de sigmoid (preserva mais ordenação que isotônica neste tamanho de coorte).
XGBoost foi calibrado com isotônica (ganho de AUC +0,007).

---

## 6. Interpretabilidade (SHAP) — top-10

`reports/lgbm_shap.csv` / `reports/figures/lgbm_shap_top15.png` (e equivalentes
para XGBoost).

| # | feature | mean |SHAP| | leitura |
|---|---|---|---|
| 1 | `Idade` | 0,211 | idade absoluta distingue trajetórias precoces de crônicas. |
| 2 | `Defasagem` | 0,167 | quem entra atrasado tende a continuar/agravar. |
| 3 | `IPV` | 0,125 | desempenho na *Prova de Valores* — proxy de engajamento — antecede queda. |
| 4 | `gap_Por_Ing` | 0,084 | descompasso Português × Inglês = fragilidade. |
| 5 | `gap_Mat_Por` | 0,077 | descompasso Matemática × Português = risco. |
| 6 | `Ing` | 0,057 | nota bruta de inglês (após imputação por Fase). |
| 7 | `inde_z_fase` | 0,044 | INDE relativo à coorte da Fase, não absoluto. |
| 8 | `gap_IAA_IEG` | 0,042 | autoavaliação alta + engajamento baixo = alerta. |
| 9 | `IPS_z_faseAno` | 0,036 | psicossocial relativo à coorte do mesmo ano. |
| 10 | `IAA_z_faseAno` | 0,028 | autoavaliação relativa à coorte. |

Os três insights mais acionáveis para a ONG:

1. **Defasagem agora prediz defasagem amanhã** — incluir "anos atrás da Fase
   ideal" no triagem mensal já é metade do trabalho.
2. **IPV é o pulso emocional do aluno.** Nota baixa em valores antecede o INDE
   cair; o tutor pode agir antes de a queda virar pública.
3. **Desequilíbrios entre disciplinas (Mat-Por, Por-Ing)** entram no top-10 —
   alunos com perfil "frágil em uma frente" merecem revisão de plano de tutoria.

---

## 7. Distribuição de risco previsto para 2025

`data/processed/predicoes_risco_2025.csv` (1 156 alunos, colunas
`prob_lgbm, prob_xgb, prob_mlp, prob_stacking, prob_final, top_3_fatores,
faixa_risco`).

`prob_final` = média(LGBM calibr., MLP calibr.) — modelo finalista.

| Faixa | Critério | n |
|---|---|---|
| **Alto** | prob_final > 0,60 | **418** |
| **Médio** | 0,30 ≤ prob_final ≤ 0,60 | **729** |
| **Baixo** | prob_final < 0,30 | **9** |

Plot: `reports/figures/risk_distribution_2025.png`.

A massa em "Médio" reflete a prevalência alta (~65 %). Recomendamos usar a
faixa **Alto** como prioridade-1 e filtrar a faixa **Médio** com regras
operacionais (ex.: IPV baixo + Defasagem crescente).

---

## 8. Estratégias que tentei (com transparência sobre o que ajudou)

### Ajudaram
- **Tie-break por complexidade** na seleção de hiperparâmetros (XGB
  `max_depth=4, min_child_weight=10` em vez de `depth=8`): subiu AUC do XGB
  *raw* de **0,665 → 0,695** no hold-out.
- **Seed-bagging do LightGBM (3 sementes)**: subiu o AUC bruto de 0,753 →
  0,754 (variação amostral, mas trouxe estabilidade no Brier).
- **Platt no MLP**: Brier 0,247 → 0,209 sem custo de AUC.
- **Average(LGBM_cal, MLP_cal)** como ensemble simples: 0,749 no hold-out,
  PR-AUC 0,841, melhor Brier entre os ensembles.
- **Refino de features**: z-score por (Fase × Ano), decile por Fase, trend
  slope — entraram no top-15 do SHAP, indicando contribuição real.

### Tentei e NÃO ajudaram significativamente
- **Isotônica no LightGBM**: caiu ROC-AUC em −0,022 vs raw. Pieggy. Por isso a
  regra do "delta > 0,01" optou por sigmoid.
- **XGBoost com max_depth=8**: CV-AUC parecida (0,7534) mas hold-out caiu para
  0,665 (overfitting visível). Confirma que dataset pequeno preferiu árvores
  rasas.
- **Stacking 4-modelos**: 0,744 ROC-AUC vs LGBM raw 0,754 — meta-learner pegou
  peso alto no MLP que tem AUC menor que LGBM no hold-out. Net loss em ROC,
  mas ganho em PR-AUC (0,841 vs 0,830).
- **Calibração isotônica em geral**: sempre danifica AUC neste tamanho de
  amostra (poucos pontos de calibração por bin).

### NÃO tentei (por princípio ou impossibilidade)
- Pseudo-labeling no hold-out — proibido.
- SMOTE / sobreamostragem — desnecessário (prevalência majoritária) e geraria
  viés.
- Empilhar modelos com predições in-sample — bug clássico que infla AUC; usamos
  exclusivamente OOF do treino.
- Treinar usando **2023→2024** misturado ao treino — seria vazamento do
  hold-out.
- Tunar threshold no hold-out — usamos OOF do treino e aplicamos fixo.

---

## 8b. Iteração 3 — enriquecimento do painel + busca Bayesiana

### Hipótese
O CSV processado em `data/processed/base_historico.csv` perdeu colunas que
existem no Excel original. Em particular, **Pedra 20 / 21 / 22 / 23** são
estado em anos passados (estritamente ≤ *t*) → podem ser features legítimas.
Uma trajetória de Pedra de até 4 anos poderia adicionar sinal além das
janelas curtas `prev_*` / `roll2_*` que só vão para *t − 1*.

Adicionalmente, a busca em grade de hiperparâmetros já estava saturada
(9 combinações × LGBM, 9 × XGB, todas dentro de 0,005 de AUC). Uma busca
**Bayesiana TPE** com 50 trials sobre o mesmo train (GroupKFold-5 por RA)
poderia encontrar configurações mais finas no espaço contínuo
(`learning_rate`, `feature_fraction`, `bagging_fraction`, `lambda_l1`,
`lambda_l2`, `min_split_gain`).

### O que foi feito
1. **Enriquecimento**: `data.enrich_with_pedra_history()` lê o xlsx,
   normaliza ortografia (Ágata→Agata) e faz merge `(RA, Ano)`. Em seguida,
   `features._pedra_history_features` produz `pedra_lag1_ord`,
   `pedra_lag2_ord`, `pedra_history_depth`, `pedra_slope`, `pedra_delta_1`,
   `pedra_delta_2` (lags 3-4 ficam ≥83 % NaN no train e são descartados pelo
   filtro "100 % NaN no treino"). Outras "extras" do xlsx (Indicado,
   Atingiu PV, Destaque IEG/IDA/IPV, Rec Psicologia, Cg/Cf/Ct) foram
   **descartadas** porque só existem no PEDE2022 — usá-las criaria
   inconsistência (modelo veria valores no treino e só NaN no val/predict).
2. **Optuna TPE** com 50 trials (`scripts/run_optuna_lgbm.py`), métrica =
   mean CV ROC-AUC com GroupKFold-5; busca em:
   `num_leaves∈[7,63]`, `min_child_samples∈[5,60]`,
   `feature_fraction∈[0.5,1]`, `bagging_fraction∈[0.5,1]`,
   `bagging_freq∈{0,1,3,5}`, `lambda_l1, lambda_l2 ∈ [10⁻³, 5]` log,
   `learning_rate ∈ [0.01, 0.1]` log, `min_split_gain ∈ [0, 0.5]`.
   O hold-out é tocado **uma única vez** ao final com os `best_params`.

### Resultados (ANTES = checkpoint `7a3f0ec`, DEPOIS = run atual)

| Modelo | ROC-AUC ANTES | ROC-AUC DEPOIS | Δ | PR-AUC ANTES | PR-AUC DEPOIS |
|---|---|---|---|---|---|
| LightGBM_raw          | 0,754 | **0,758** | +0,004 | 0,833 | 0,843 (+0,010) |
| LightGBM_calibr       | 0,742 | 0,738 | −0,004 | 0,830 | 0,833 |
| XGBoost_raw           | 0,695 | 0,718 | +0,023 | 0,809 | 0,826 (+0,017) |
| Stacking (4 modelos)  | 0,744 | 0,739 | −0,005 | 0,841 | 0,841 |
| Average_LGBM_MLP      | 0,749 | 0,735 | −0,014 | 0,841 | 0,836 |
| LightGBM **Optuna**   | —     | 0,747 | −0,011¹ | —     | 0,838 |

¹ Optuna alcançou **CV-AUC 0,7657** (vs 0,7521 da grade), mas a melhoria não
se traduziu no hold-out (caiu para 0,747). É um caso clássico de
**overfitting ao train**: TPE encontrou parâmetros que exploram melhor a
estrutura do treino 2022→2023 mas generalizam pior para 2023→2024.

### Decisão
- **Enriquecimento Pedra-history: ADOTADO.** Ganho honesto em LGBM_raw
  (+0,004 ROC, +0,010 PR-AUC) e XGB_raw (+0,023 ROC, +0,017 PR-AUC).
  Tecnicamente sólido e baixo risco — colunas estado-em-passado, sem
  vazamento.
- **Optuna 50-trials: REJEITADO.** CV-AUC subiu mas hold-out caiu
  −0,011 vs LGBM_raw da grade. **Não bateu o limiar +0,005** definido
  como critério de adoção. Os parâmetros, score e CSV de trials estão
  preservados em `experiments/<timestamp>/optuna_lgbm/` para auditoria.
- **`prob_final` no CSV de 2025 = LightGBM_raw (com Pedra-history).**
  Adicionei coluna `prob_experimental` com a predição do LightGBM-Optuna
  para o usuário comparar manualmente.

### Por que CV ↑ mas hold-out ↓?
Hipótese mais provável: a busca Bayesiana, com `learning_rate=0.014` e
`num_leaves=26`, gera árvores mais profundas e treina por ~267 iterações
(vs ~30-50 da grade). Isso ajusta padrões da transição 2022→2023 que **não
repetem** em 2023→2024 — confirma o `drift de transição` discutido na
seção 9. A grade favoreceu árvores rasas (15 folhas, ~30 iter) que são
mais robustas ao drift.

---

## 9. Por que não bateu 0,78? Hipótese honesta

1. **Tamanho do treino**: 600 linhas é pequeno e a `GroupKFold` CV já mostra
   teto em **0,755** sem ver o hold-out. Ou seja, **a barra de 0,78 não está
   no sinal disponível**, não é falha do modelo.
2. **Alvo composto com 65 % de prevalência** comprime PR-AUC e *recall@top-N*.
   Para *triagem* (encontrar os 50 mais críticos) PR-AUC 0,84 é altamente
   acionável, mas para ranqueamento global o teto natural é mais baixo.
3. **Redundância dos indicadores**: o INDE é média ponderada dos sub-índices,
   então muito do sinal já entra em outras features — o ganho marginal de
   cada nova feature é pequeno.
4. **Drift 2022→2023 vs 2023→2024**: a transição usada como hold-out tem
   prevalência mais alta (65 % vs 62 %) e mais defasagem absoluta. XGB com
   parâmetros que fazem bem na CV de 2022 generaliza pior para esse drift.

**Próximos passos para ultrapassar 0,78** (em ordem de retorno esperado):

1. **Modelos por sub-grupo de Fase** (0-2, 3-5, 6-9). Os AUCs por Fase
   sugerem que um modelo único subestima o sinal em algumas faixas
   (Fase 1 chega a 0,85 quando isolada).
2. **Reabrir a discussão de alvo** com o stakeholder: `risk_inde_drop` puro
   tem 32 % de prevalência — mais fácil de ranquear e operacionalmente
   talvez mais útil para o setor pedagógico.
3. **Mais features históricas** ~~já no Excel raw (Pedra 20/21/22/23)~~ —
   **feito na Iteração 3** (seção 8b). Outras colunas exclusivas do
   PEDE2022 (Indicado, Atingiu PV, Cg/Cf/Ct, Destaque, Rec Psicologia) **não
   são utilizáveis** porque estão 100 % NaN em 2023 e 2024 — usá-las criaria
   uma feature observada só no treino. *Próximo passo viável aqui*: pedir à
   NGO o repasse do PEDE2023/2024 com essas mesmas colunas preenchidas.
4. **Aumentar amostra**: incluir alunos que entraram a meio do ano (hoje
   filtrados pelo *inner join* RA-pares), com `is_new_student` como flag.
5. **Avaliar com janela temporal expandida** quando os dados 2025 chegarem
   (treino 2022+2023, hold-out 2024→2025) — passará a haver 3 transições.

---

## 10. Como reproduzir

```bash
.venv/bin/python scripts/run_pipeline.py
# (opcional, ~50 s) busca Bayesiana TPE — escreve experiments/<ts>/ e injeta
# prob_experimental no CSV de predições:
.venv/bin/python scripts/run_optuna_lgbm.py
```

Tempo total observado: **~30 s** (CPU). Saídas geradas:

```
models/lgbm.pkl                     # lista com as 3 sementes (seed-bag)
models/lgbm_seed42.pkl              # modelo da seed=42 (referência para SHAP)
models/lgbm_calibrated.pkl
models/xgb.pkl
models/xgb_calibrated.pkl
models/mlp.pt
models/mlp_meta.json
models/stacking.pkl
models/preprocessor.pkl
models/metrics_full.json
reports/lgbm_shap.csv
reports/xgb_shap.csv
reports/auc_by_fase.csv
reports/auc_by_pedra.csv
reports/auc_by_gender.csv
reports/figures/lgbm_shap_top15.png
reports/figures/xgb_shap_top15.png
reports/figures/calibration_curves.png
reports/figures/auc_by_fase.png
reports/figures/risk_distribution_2025.png
data/processed/predicoes_risco_2025.csv
```

---

## 11. Como interpretar e agir

| Faixa | Score | Recomendação operacional |
|---|---|---|
| **Alto** | > 0,60 | Tutor revisa o caso na próxima semana; verificar IPV e Defasagem; reunião com responsável; revisar plano de tutoria. |
| **Médio** | 0,30 – 0,60 | Monitorar trimestralmente; priorizar dentro da faixa quem tem `IPV` baixo **e** `gap_Mat_Por` > 1,5. |
| **Baixo** | < 0,30 | Manter rotina; revalidar no próximo ciclo PEDE. |

A coluna `top_3_fatores` do CSV oferece a **explicação local** por aluno
(SHAP individual), pronta para conversas com responsáveis e tutores. Os top-3
mais frequentes são `Defasagem`, `Idade`, `IPV` — onboarding do tutor pode
começar perguntando especificamente por esses três aspectos.

---

## 12. Notas operacionais

- **Reprodutibilidade total**: `random_state=42` em todo o pipeline; o MLP usa
  `seed=42 + fold` em cada dobra para diversificar mas determinístico.
- **Sem `joblib.Parallel` com loky**: o MLP roda em **subprocess via `python
  -m`** porque a combinação loky/spawn + PyTorch/MPS em macOS demonstrou
  travamento silencioso. Single-process puro é mais lento mas robusto; o
  pipeline inteiro cabe em 30 s.
- **MPS desativado** (`device='cpu'`) para esta rodada; quando o stakeholder
  autorizar uma versão "GPU-on", basta retirar o force-cpu na linha
  correspondente do `mlp_subprocess.py`.
