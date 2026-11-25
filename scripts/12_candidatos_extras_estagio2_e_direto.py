"""
Candidatos extras no estagio 2 e no multiclasse direto (pendencia 4 da secao "Pendencias" de
../docs/DECISOES-E-BIBLIOGRAFIA.md): Extra Trees, XGBoost e kNN, que o 06 e o 07 nao testaram.
Mesmo protocolo (K=5, mesmos folds, F1 macro, McNemar) e salvando em
resultados/metricas/estagio2_extras/ e multiclasse_direto_extras/. O teste nao e tocado.

- Estagio 2: mesmo balanceamento do RF escolhido (SMOTE ate 5 mil fluxos nas classes raras) para
  Extra Trees e XGBoost. O McNemar compara com o RF+SMOTE do 06 (previsao final em 15 classes).
- Multiclasse direto: mesmo balanceamento do RF escolhido (pesos balanceados). O McNemar compara
  com o RF com class_weight do 07.
- kNN: subamostra de no maximo 20 mil fluxos por classe em cada fold (Aula 4, "Amostragem de
  dados"), k = 5, como no estagio 1 (script 11).
- XGBoost exige rotulos 0..n-1 e nao tem class_weight: o envoltorio abaixo recodifica os rotulos
  e aplica pesos balanceados (n / (n_classes * n_da_classe)), a mesma formula do class_weight
  "balanced" do scikit-learn.
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import modelagem_comum as mc
import importlib
est2 = importlib.import_module("06_estagio2_tipo_ataque")

S = mc.SEMENTE
TETO_KNN = 20_000


class XGBMulticlasse(BaseEstimator, ClassifierMixin):
    def __init__(self, balanceado=False, n_estimators=200):
        self.balanceado = balanceado
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.classes_, y_cod = np.unique(y, return_inverse=True)
        peso = None
        if self.balanceado:
            contagem = np.bincount(y_cod)
            peso = (len(y_cod) / (len(contagem) * contagem))[y_cod]
        self.modelo_ = XGBClassifier(n_estimators=self.n_estimators, tree_method="hist",
                                     n_jobs=-1, random_state=S)
        self.modelo_.fit(X, y_cod, sample_weight=peso)
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)

    def predict(self, X):
        return self.classes_[self.modelo_.predict(X)]


def teto_por_classe(y):
    return {c: min(n, TETO_KNN) for c, n in Counter(y).items()}


def knn():
    return Pipeline([
        ("under", RandomUnderSampler(sampling_strategy=teto_por_classe, random_state=S)),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ])


def smote():
    return SMOTE(sampling_strategy=est2.alvo_smote, k_neighbors=5, random_state=S)


CAND_ESTAGIO2 = {
    "et_smote": Pipeline([("smote", smote()),
                          ("et", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=S))]),
    "xgb_smote": Pipeline([("smote", smote()), ("xgb", XGBMulticlasse())]),
    "knn_subamostra": knn(),
}

CAND_DIRETO = {
    "et_class_weight": ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=S,
                                            class_weight="balanced"),
    "xgb_pesos_balanceados": XGBMulticlasse(balanceado=True),
    "knn_subamostra": knn(),
}


def rodar_estagio2(X, y_multi, nomes, folds):
    etapa = "estagio2_extras"
    (mc.MET_DIR / etapa).mkdir(parents=True, exist_ok=True)
    (mc.PREV_DIR / etapa).mkdir(parents=True, exist_ok=True)
    classes = np.unique(y_multi.to_numpy())
    est1 = est2.escolher_estagio1()
    oof1 = pd.read_parquet(mc.PREV_DIR / "estagio1" / f"{est1}_oof.parquet")
    pred1 = oof1["y_pred"].to_numpy()
    base = pd.read_parquet(mc.PREV_DIR / "estagio2" / "rf_smote_oof.parquet")
    finais = {"rf_smote": base["y_pred_final"].to_numpy()}
    medias = {"rf_smote": pd.read_csv(mc.MET_DIR / "estagio2" / "rf_smote_resumo.csv", index_col=0)["media"]}
    for nome, modelo in CAND_ESTAGIO2.items():
        print(f"\n[estagio 2] {nome}", flush=True)
        pred2, fold_de, por_fold = est2.validar_estagio2(modelo, X, y_multi, pred1, folds)
        resumo = mc.resumir_folds(por_fold)
        por_fold.to_csv(mc.MET_DIR / etapa / f"{nome}_por_fold.csv")
        resumo.to_csv(mc.MET_DIR / etapa / f"{nome}_resumo.csv")
        final = np.where(pred1 == 0, 0, pred2)
        pd.DataFrame({"fold": fold_de, "y_true": y_multi.to_numpy(), "pred_estagio2": pred2,
                      "y_pred_final": final}).to_parquet(mc.PREV_DIR / etapa / f"{nome}_oof.parquet", index=False)
        mc.metricas_por_classe(y_multi.to_numpy(), final, classes, nomes).to_csv(
            mc.MET_DIR / etapa / f"{nome}_por_classe_final.csv")
        finais[nome] = final
        medias[nome] = resumo["media"]
        print(resumo.loc[["f1_macro_estagio2", "f1_macro_final", "acuracia_final", "tempo_treino_s"], :].round(4).to_string(), flush=True)
    comp = pd.DataFrame(medias).T.sort_values("f1_macro_final", ascending=False)
    comp.to_csv(mc.MET_DIR / etapa / "comparativo_candidatos.csv")
    print("\n[estagio 2] comparativo:")
    print(comp[["f1_macro_estagio2", "f1_macro_final", "acuracia_final", "tempo_treino_s", "tempo_previsao_s"]].round(4).to_string(), flush=True)
    mc.mcnemar_pares(y_multi, finais).to_csv(mc.MET_DIR / etapa / "mcnemar_pares.csv", index=False)


def rodar_direto(X, y_multi, nomes, folds):
    etapa = "multiclasse_direto_extras"
    classes = np.unique(y_multi.to_numpy())
    base = pd.read_parquet(mc.PREV_DIR / "multiclasse_direto" / "rf_class_weight_oof.parquet")
    previsoes = {"rf_class_weight": base["y_pred"].to_numpy()}
    medias = {"rf_class_weight": pd.read_csv(mc.MET_DIR / "multiclasse_direto" / "rf_class_weight_resumo.csv", index_col=0)["media"]}
    for nome, modelo in CAND_DIRETO.items():
        print(f"\n[direto] {nome}", flush=True)
        pred, score, fold_de, por_fold = mc.validacao_cruzada(
            modelo, X, y_multi, folds, lambda y, p, s: mc.metricas_multiclasse(y, p, s, classes=classes))
        resumo = mc.resumir_folds(por_fold)
        mc.salvar_validacao(etapa, nome, por_fold, resumo, y_multi, pred, score, fold_de)
        mc.metricas_por_classe(y_multi.to_numpy(), pred, classes, nomes).to_csv(
            mc.MET_DIR / etapa / f"{nome}_por_classe.csv")
        previsoes[nome] = pred
        medias[nome] = resumo["media"]
        print(resumo.round(4).to_string(), flush=True)
    comp = pd.DataFrame(medias).T.sort_values("f1_macro", ascending=False)
    comp.to_csv(mc.MET_DIR / etapa / "comparativo_candidatos.csv")
    print("\n[direto] comparativo:")
    print(comp[["f1_macro", "acuracia", "revocacao_macro", "tempo_treino_s", "tempo_previsao_s"]].round(4).to_string(), flush=True)
    mc.mcnemar_pares(y_multi, previsoes).to_csv(mc.MET_DIR / etapa / "mcnemar_pares.csv", index=False)


if __name__ == "__main__":
    parte = sys.argv[1] if len(sys.argv) > 1 else "tudo"
    X, y_bin, y_multi = mc.carregar("train")
    nomes = mc.nomes_classes()
    folds = mc.folds_estratificados(y_multi)
    if parte in ("estagio2", "tudo"):
        rodar_estagio2(X, y_multi, nomes, folds)
    if parte in ("direto", "tudo"):
        rodar_direto(X, y_multi, nomes, folds)
