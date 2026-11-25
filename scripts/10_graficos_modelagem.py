"""
Figuras da modelagem para os slides e o relatorio (le so os CSV dos scripts 05 a 09).
Saida: ../resultados/figuras/modelagem_*.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import modelagem_comum as mc

FIG = mc.BASE_DIR / "resultados" / "figuras"
MET = mc.MET_DIR
AZUL, LARANJA, CINZA = "#2c6e9b", "#d98324", "#8a8a8a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

ROTULOS = {
    "rf_sem_balanceamento": "RF sem balanceamento",
    "rf_class_weight": "RF class_weight",
    "rf_undersampling": "RF undersampling",
    "rf_smote": "RF SMOTE",
}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(FIG / nome, dpi=160)
    plt.close(fig)
    print("salvo", nome)


# 1. Estagio 1: candidatos (validacao cruzada); laranja = o escolhido

comp = pd.read_csv(MET / "estagio1" / "comparativo_candidatos.csv", index_col=0).drop(index="baseline_majoritaria")
sd = pd.concat([pd.read_csv(MET / "estagio1" / f"{n}_resumo.csv", index_col=0)["desvio_padrao"].rename(n)
                for n in comp.index], axis=1).T
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, (col, titulo) in zip(axes, [("f1", "F1 (ataque)"), ("revocacao", "Revocação"), ("tempo_treino_s", "Tempo de treino por fold (s)")]):
    ax.bar([ROTULOS[i] for i in comp.index], comp[col], yerr=sd[col], color=[LARANJA if i == comp["f1"].idxmax() else AZUL for i in comp.index], capsize=3)
    ax.set_title(titulo)
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    if col != "tempo_treino_s":
        ax.set_ylim(comp[col].min() - 0.004, 1.0005)
salvar(fig, "modelagem_estagio1_candidatos.png")

# 2. Matriz de confusao do hierarquico no teste (normalizada por linha)
cm = pd.read_csv(MET / "final" / "hierarquico_matriz_confusao_teste.csv", index_col=0)
norm = cm.div(cm.sum(axis=1), axis=0)
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=60, ha="right")
ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index)
ax.set_xlabel("Classe prevista"); ax.set_ylabel("Classe real")
for i in range(len(cm)):
    for j in range(len(cm)):
        v = norm.values[i, j]
        if v >= 0.005:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.5 else "black")
ax.set_title("Pipeline hierárquico no teste (proporção de cada classe real)")
fig.colorbar(im, ax=ax, fraction=0.04)
salvar(fig, "modelagem_matriz_confusao_hierarquico.png")

# 3. F1 por classe: hierarquico x direto, com o suporte do teste
h = pd.read_csv(MET / "final" / "hierarquico_por_classe_teste.csv", index_col=0)
d = pd.read_csv(MET / "final" / "multiclasse_direto_por_classe_teste.csv", index_col=0)
ordem = h.sort_values("suporte", ascending=False).index
y = np.arange(len(ordem))
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(y - 0.2, h.loc[ordem, "f1"], height=0.4, color=AZUL, label="Hierárquico")
ax.barh(y + 0.2, d.loc[ordem, "f1"], height=0.4, color=LARANJA, label="Multiclasse direto")
ax.set_yticks(y)
ax.set_yticklabels([f"{h.loc[c, 'nome']} (n={int(h.loc[c, 'suporte']):,})".replace(",", ".") for c in ordem])
ax.invert_yaxis()
ax.set_xlim(0, 1.05)
ax.set_xlabel("F1 por classe no teste")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=2, frameon=False)
ax.set_title("Desempenho por classe (ordenado pelo suporte)")
salvar(fig, "modelagem_f1_por_classe.png")

# 4. Taxa de deteccao do estagio 1 por tipo de ataque (onde o detector perde fluxos)
t = pd.read_csv(MET / "final" / "estagio1_deteccao_por_tipo_teste.csv")
t = t[t["Label_multiclasse"] != 0].sort_values("taxa_deteccao")
fig, ax = plt.subplots(figsize=(7, 4.8))
ax.barh(t["nome"], t["taxa_deteccao"], color=[LARANJA if v < 0.99 else AZUL for v in t["taxa_deteccao"]])
ax.set_xlim(0, 1.02)
ax.set_xlabel("Fração dos fluxos detectada como ataque (estágio 1, teste)")
for i, (v, n) in enumerate(zip(t["taxa_deteccao"], t["suporte"])):
    ax.text(min(v, 1) - 0.01, i, f"{v:.2f} (n={n})", va="center", ha="right", fontsize=8,
            color="white" if v > 0.3 else "black")
ax.set_title("Detecção por tipo de ataque")
salvar(fig, "modelagem_deteccao_por_tipo.png")

# 5. SHAP x EDA: importancia SHAP e correlacao com o alvo, mesmos atributos
imp = pd.read_csv(MET / "xai" / "shap_global_estagio1.csv", index_col=0).iloc[:, 0]
corr = pd.read_csv(MET / "correlacao_com_alvo.csv", index_col=0).iloc[:, 0].abs().dropna()
comum = imp.index.intersection(corr.index)
rho = pd.read_csv(MET / "xai" / "shap_x_eda.csv").iloc[0]["spearman"]
fig, ax = plt.subplots(figsize=(6, 4.6))
ax.scatter(corr[comum], imp[comum], color=AZUL, s=22)
for a in imp.head(6).index.intersection(comum):
    ax.annotate(a, (corr[a], imp[a]), fontsize=7, xytext=(4, 3), textcoords="offset points")
ax.set_xlabel("|correlação com o alvo| (EDA)")
ax.set_ylabel("média do |SHAP| (estágio 1)")
ax.set_title(f"SHAP x EDA (Spearman = {rho:.2f})")
salvar(fig, "modelagem_shap_x_eda.png")
