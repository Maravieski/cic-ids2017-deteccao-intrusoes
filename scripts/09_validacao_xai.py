# %% [markdown]
# # Fase 3: validacao com XAI (SHAP)
# Secao 14 de ../docs/DECISOES-E-BIBLIOGRAFIA.md. Pergunta central: o modelo aprendeu
# padrao de ataque ou artefato do laboratorio (shortcut learning, decisao 2)?
#
# Usa os modelos salvos pelo 08. TreeSHAP sobre uma amostra estratificada do
# conjunto de teste (todas as classes raras inteiras, no maximo N_POR_CLASSE
# fluxos das demais), nao o teste inteiro.
# Formato "percent": cada `# %%` e uma celula, abre como notebook no VS Code/Jupytext.

# %%
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import modelagem_comum as mc

N_POR_CLASSE = 300
TOP_LOCAL = 5
FIG_DIR = mc.BASE_DIR / "resultados" / "figuras"
SAIDA = mc.MET_DIR / "xai"
SAIDA.mkdir(parents=True, exist_ok=True)

X_te, yb_te, ym_te = mc.carregar("test")
nomes = mc.nomes_classes()
est1 = joblib.load(mc.BASE_DIR / "dados" / "modelos" / "estagio1.joblib")["modelo"]
est2 = joblib.load(mc.BASE_DIR / "dados" / "modelos" / "estagio2.joblib")["modelo"]

# %% [markdown]
# ## Amostra estratificada

# %%
rng = np.random.default_rng(mc.SEMENTE)
idx = []
for classe, grupo in ym_te.groupby(ym_te):
    pos = grupo.index.to_numpy()
    idx.extend(pos if len(pos) <= N_POR_CLASSE else rng.choice(pos, N_POR_CLASSE, replace=False))
amostra = X_te.loc[sorted(idx)]
y_amostra = ym_te.loc[amostra.index]
print(f"Amostra: {len(amostra)} fluxos de {len(X_te):,}")


def valores_shap(modelo, X, classe_alvo=None):
    """TreeSHAP; devolve array (n, atributos) de uma classe. Cobre os dois formatos de saida do shap."""
    bruto = shap.TreeExplainer(modelo).shap_values(X, check_additivity=False)
    if isinstance(bruto, list):
        bruto = np.stack(bruto, axis=-1)
    if bruto.ndim == 3:
        j = 1 if classe_alvo is None else list(modelo.classes_).index(classe_alvo)
        return bruto[:, :, j]
    return bruto


# %% [markdown]
# ## SHAP global do estagio 1 (BENIGN x ataque), comparado com a EDA

# %%
sv1 = valores_shap(est1, amostra)
importancia = pd.Series(np.abs(sv1).mean(axis=0), index=amostra.columns).sort_values(ascending=False)
importancia.to_csv(SAIDA / "shap_global_estagio1.csv", header=["media_abs_shap"])

corr = pd.read_csv(mc.MET_DIR / "correlacao_com_alvo.csv", index_col=0).iloc[:, 0]
enr = pd.read_csv(mc.MET_DIR / "outliers_enriquecimento_ataque.csv", index_col=0)["enriquecimento"]
corr = corr.dropna()
enr = enr.dropna()
comum = importancia.index.intersection(corr.index)
rho_corr = spearmanr(importancia[comum], corr[comum].abs())
comum_e = importancia.index.intersection(enr.index)
rho_enr = spearmanr(importancia[comum_e], enr[comum_e])
resumo = pd.DataFrame({
    "comparacao": ["SHAP x |correlacao com o alvo|", "SHAP x enriquecimento de outliers"],
    "n_atributos": [len(comum), len(comum_e)],
    "spearman": [rho_corr.statistic, rho_enr.statistic],
    "p_valor": [rho_corr.pvalue, rho_enr.pvalue],
})
resumo.to_csv(SAIDA / "shap_x_eda.csv", index=False)
print(resumo.to_string(index=False))
print("\nTop 15 atributos (SHAP, estagio 1):")
print(importancia.head(15).round(4).to_string())

topo = importancia.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(topo.index, topo.values, color="#2c6e9b")
ax.set_xlabel("media do |valor SHAP| (estagio 1)")
ax.set_title("Atributos mais importantes para detectar ataque")
fig.tight_layout()
fig.savefig(FIG_DIR / "shap_global_estagio1.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## SHAP global do estagio 2, por classe de ataque

# %%
ataques = y_amostra[y_amostra != 0]
X_ataques = amostra.loc[ataques.index]
bruto2 = shap.TreeExplainer(est2).shap_values(X_ataques.to_numpy(), check_additivity=False)
if isinstance(bruto2, list):
    bruto2 = np.stack(bruto2, axis=-1)
classes2 = list(est2.classes_)
linhas = []
for j, classe in enumerate(classes2):
    imp = pd.Series(np.abs(bruto2[:, :, j]).mean(axis=0), index=X_ataques.columns)
    for rank, (atributo, valor) in enumerate(imp.sort_values(ascending=False).head(5).items(), start=1):
        linhas.append({"classe": nomes[int(classe)], "rank": rank, "atributo": atributo, "media_abs_shap": valor})
pd.DataFrame(linhas).to_csv(SAIDA / "shap_global_estagio2_top5_por_classe.csv", index=False)
print("\nTop 3 por classe (estagio 2):")
print(pd.DataFrame(linhas).query("rank <= 3").to_string(index=False))

# %% [markdown]
# ## Explicacao local por fluxo (entrada do RAG)
# Para cada fluxo da amostra que o estagio 1 marca como ataque: classe prevista pelo estagio 2
# e os TOP_LOCAL atributos que mais empurraram a decisao do estagio 1 (com o valor padronizado).

# %%
pred1 = est1.predict(amostra.to_numpy())
marcados = np.where(pred1 == 1)[0]
pred2 = est2.predict(amostra.to_numpy()[marcados])
linhas = []
for k, pos in enumerate(marcados):
    contrib = pd.Series(sv1[pos], index=amostra.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(TOP_LOCAL)
    linhas.append({
        "indice_original": int(amostra.index[pos]),
        "classe_real": nomes[int(y_amostra.iloc[pos])],
        "classe_prevista": nomes[int(pred2[k])],
        "atributos_top": "; ".join(f"{a} ({amostra.iloc[pos][a]:+.2f}, shap {v:+.3f})" for a, v in top.items()),
    })
local = pd.DataFrame(linhas)
local.to_csv(SAIDA / "explicacoes_locais.csv", index=False)
print(f"\n{len(local)} explicacoes locais salvas. Exemplo:")
print(local.sample(3, random_state=mc.SEMENTE).to_string(index=False))

# %% [markdown]
# ## Analise dos erros

# %%
final = np.where(pred1 == 0, 0, est2.predict(amostra.to_numpy()))
erros = pd.DataFrame({"classe_real": [nomes[int(c)] for c in y_amostra],
                      "classe_prevista": [nomes[int(c)] for c in final]})
erros = erros[erros.classe_real != erros.classe_prevista]
tabela = erros.value_counts().rename("n").reset_index()
tabela.to_csv(SAIDA / "erros_amostra.csv", index=False)
print(f"\nErros na amostra: {len(erros)} de {len(amostra)}")
print(tabela.head(15).to_string(index=False))
