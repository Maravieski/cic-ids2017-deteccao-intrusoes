"""
Avaliacao final no conjunto de teste (secao 10 de ../docs/DECISOES-E-BIBLIOGRAFIA.md):
o teste so entra aqui, uma unica vez, nos modelos ja escolhidos pelo K-fold.

Escolhas (pelo comparativo salvo por cada script, sem olhar o teste):
- estagio 1: maior F1 da classe ataque (05, sem a baseline);
- estagio 2: maior F1 macro da previsao final em 15 classes (06);
- multiclasse direto: maior F1 macro (07).

Cada modelo e reajustado com o treino inteiro. Salva os modelos em
../dados/modelos/ (entrada do SHAP) e as previsoes do teste em
../resultados/previsoes/final/.
"""
import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

import modelagem_comum as mc

ETAPA = "final"
MODELOS_DIR = mc.BASE_DIR / "dados" / "modelos"


def carregar_script(arquivo: str):
    spec = importlib.util.spec_from_file_location(arquivo, Path(__file__).parent / f"{arquivo}.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def so_classificador(modelo):
    """Tira o balanceamento do Pipeline (so atua no fit) e devolve o classificador, que e o que o SHAP e o RAG usam."""
    return modelo.steps[-1][1] if hasattr(modelo, "steps") else modelo


def melhor(etapa: str, coluna: str) -> str:
    comp = pd.read_csv(mc.MET_DIR / etapa / "comparativo_candidatos.csv", index_col=0)
    comp = comp.drop(index=[i for i in comp.index if i.startswith("baseline")])
    return comp[coluna].idxmax()


if __name__ == "__main__":
    s5 = carregar_script("05_estagio1_binario")
    s6 = carregar_script("06_estagio2_tipo_ataque")
    s7 = carregar_script("07_multiclasse_direto")

    nome1 = melhor("estagio1", "f1")
    nome2 = melhor("estagio2", "f1_macro_final")
    nome7 = melhor("multiclasse_direto", "f1_macro")
    print(f"Escolhidos: estagio 1 = {nome1}, estagio 2 = {nome2}, multiclasse direto = {nome7}")

    X_tr, yb_tr, ym_tr = mc.carregar("train")
    X_te, yb_te, ym_te = mc.carregar("test")
    nomes = mc.nomes_classes()
    classes = np.unique(ym_tr.to_numpy())
    (mc.MET_DIR / ETAPA).mkdir(parents=True, exist_ok=True)
    (mc.PREV_DIR / ETAPA).mkdir(parents=True, exist_ok=True)
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)

    # estagio 1
    m1 = clone(s5.CANDIDATOS[nome1]).fit(X_tr.to_numpy(), yb_tr.to_numpy())
    score1 = m1.predict_proba(X_te.to_numpy())
    pred1 = m1.predict(X_te.to_numpy())
    met1 = mc.metricas_binarias(yb_te, pred1, score1)
    pd.Series(met1).to_csv(mc.MET_DIR / ETAPA / "estagio1_teste.csv", header=["valor"])
    por_tipo = s5.deteccao_por_tipo(ym_te, pred1, nomes)
    por_tipo.to_csv(mc.MET_DIR / ETAPA / "estagio1_deteccao_por_tipo_teste.csv")
    print("\nEstagio 1 no teste:", {k: round(float(v), 4) for k, v in met1.items()})

    # estagio 2 (treina so nos ataques do treino)
    ataque_tr = ym_tr.to_numpy() != 0
    m2 = clone(s6.CANDIDATOS[nome2]).fit(X_tr.to_numpy()[ataque_tr], ym_tr.to_numpy()[ataque_tr])
    pred2 = m2.predict(X_te.to_numpy())
    final = np.where(pred1 == 0, 0, pred2)
    met_final = mc.metricas_multiclasse(ym_te, final, classes=classes)
    pd.Series(met_final).to_csv(mc.MET_DIR / ETAPA / "hierarquico_teste.csv", header=["valor"])
    mc.metricas_por_classe(ym_te.to_numpy(), final, classes, nomes).to_csv(
        mc.MET_DIR / ETAPA / "hierarquico_por_classe_teste.csv")
    pd.DataFrame(confusion_matrix(ym_te, final, labels=classes),
                 index=[nomes[c] for c in classes], columns=[nomes[c] for c in classes]).to_csv(
        mc.MET_DIR / ETAPA / "hierarquico_matriz_confusao_teste.csv")
    print("Hierarquico (15 classes) no teste:", {k: round(float(v), 4) for k, v in met_final.items()})

    # multiclasse direto
    m7 = clone(s7.CANDIDATOS[nome7]).fit(X_tr.to_numpy(), ym_tr.to_numpy())
    pred7 = m7.predict(X_te.to_numpy())
    met7 = mc.metricas_multiclasse(ym_te, pred7, classes=classes)
    pd.Series(met7).to_csv(mc.MET_DIR / ETAPA / "multiclasse_direto_teste.csv", header=["valor"])
    mc.metricas_por_classe(ym_te.to_numpy(), pred7, classes, nomes).to_csv(
        mc.MET_DIR / ETAPA / "multiclasse_direto_por_classe_teste.csv")
    print("Multiclasse direto no teste:", {k: round(float(v), 4) for k, v in met7.items()})

    # hierarquico x direto
    pares = pd.DataFrame([{"modelo_a": "hierarquico", "modelo_b": "multiclasse_direto",
                           **mc.mcnemar(ym_te, final, pred7)}])
    pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_hierarquico_x_direto_teste.csv", index=False)
    print("\nMcNemar hierarquico x direto:", pares.iloc[0].to_dict())

    pd.DataFrame({"y_true": ym_te.to_numpy(), "pred_estagio1": pred1, "score_ataque": score1[:, -1],
                  "pred_estagio2": pred2, "pred_final": final, "pred_direto": pred7}).to_parquet(
        mc.PREV_DIR / ETAPA / "teste.parquet", index=False)
    joblib.dump({"nome": nome1, "modelo": so_classificador(m1)}, MODELOS_DIR / "estagio1.joblib", compress=3)
    joblib.dump({"nome": nome2, "modelo": so_classificador(m2)}, MODELOS_DIR / "estagio2.joblib", compress=3)
    joblib.dump({"nome": nome7, "modelo": so_classificador(m7)}, MODELOS_DIR / "multiclasse_direto.joblib", compress=3)
    print(f"\nModelos salvos em {MODELOS_DIR}")
