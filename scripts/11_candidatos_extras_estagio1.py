"""
Candidatos extras no estagio 1 (secao 11 de ../docs/DECISOES-E-BIBLIOGRAFIA.md: kNN, Extra Trees
e XGBoost vem depois do Random Forest). Mesmo protocolo do 05 (K=5, mesmos folds, F1 do
ataque, McNemar), salvando em resultados/metricas/estagio1_extras/. O McNemar compara cada
candidato com o RF com undersampling do 05 (o estagio 1 escolhido), usando as previsoes
fora do fold que o 05 ja salvou. O teste nao e tocado aqui.

- Extra Trees e XGBoost: undersampling do BENIGN, como o RF escolhido (comparacao justa).
- kNN: subamostra balanceada de 50 mil fluxos de cada classe por fold (Aula 4, "Amostragem
  de dados": o kNN nao escala para 1,2 milhao de fluxos), k = 5.
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

import modelagem_comum as mc

ETAPA = "estagio1_extras"
S = mc.SEMENTE

CANDIDATOS = {
    "et_undersampling": Pipeline([
        ("under", RandomUnderSampler(random_state=S)),
        ("et", ExtraTreesClassifier(n_estimators=100, n_jobs=-1, random_state=S)),
    ]),
    "xgb_undersampling": Pipeline([
        ("under", RandomUnderSampler(random_state=S)),
        ("xgb", XGBClassifier(n_estimators=200, tree_method="hist", n_jobs=-1, random_state=S)),
    ]),
    "knn_subamostra": Pipeline([
        ("under", RandomUnderSampler(sampling_strategy={0: 50_000, 1: 50_000}, random_state=S)),
        ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ]),
}

if __name__ == "__main__":
    X, y_bin, y_multi = mc.carregar("train")
    folds = mc.folds_estratificados(y_multi)
    previsoes = {"rf_undersampling": pd.read_parquet(mc.PREV_DIR / "estagio1" / "rf_undersampling_oof.parquet")["y_pred"].to_numpy()}
    medias = {"rf_undersampling": pd.read_csv(mc.MET_DIR / "estagio1" / "rf_undersampling_resumo.csv", index_col=0)["media"]}

    for nome, modelo in CANDIDATOS.items():
        print(f"\n{nome}", flush=True)
        pred, score, fold_de, por_fold = mc.validacao_cruzada(modelo, X, y_bin, folds, mc.metricas_binarias)
        resumo = mc.resumir_folds(por_fold)
        mc.salvar_validacao(ETAPA, nome, por_fold, resumo, y_bin, pred, score, fold_de)
        previsoes[nome] = pred
        medias[nome] = resumo["media"]
        print(resumo.round(4).to_string(), flush=True)

    comparativo = pd.DataFrame(medias).T.sort_values("f1", ascending=False)
    comparativo.to_csv(mc.MET_DIR / ETAPA / "comparativo_candidatos.csv")
    print("\nComparativo (media dos folds, ordenado pelo F1):")
    print(comparativo[["f1", "acuracia", "precisao", "revocacao", "especificidade", "auc", "tempo_treino_s", "tempo_previsao_s"]].round(4).to_string())
    pares = mc.mcnemar_pares(y_bin, previsoes)
    pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_pares.csv", index=False)
    print("\nMcNemar:")
    print(pares.to_string(index=False))
