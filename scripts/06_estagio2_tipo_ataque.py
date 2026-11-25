"""
Estagio 2 do pipeline hierarquico (secao 9 de ../docs/DECISOES-E-BIBLIOGRAFIA.md):
classificador do tipo de ataque (14 classes), treinado so nos fluxos de ataque.
Testa a H2 (multiclasse mais dificil que binario) e a H3 (classes raras).

Protocolo da secao 10, com os mesmos folds do 05. Em cada fold:
- o estagio 2 treina so nos fluxos de ataque da parte de treino do fold
  (balanceamento da secao 12 como etapa do Pipeline, so no fit);
- a previsao final em 15 classes e BENIGN quando o estagio 1 diz BENIGN (previsao
  fora do fold salva pelo 05) e a classe do estagio 2 nos demais casos.

Metricas calculadas em duas visoes:
- estagio 2 isolado: so os fluxos de ataque de verdade do fold de validacao
  (mede o quanto o estagio 2 acerta o tipo quando o ataque ja foi detectado);
- previsao final: todos os fluxos do fold, com o estagio 1 na frente
  (mede o pipeline completo, que e o que o relatorio compara com o 07).

O estagio 1 usado e o candidato de maior F1 em ../resultados/metricas/estagio1/
comparativo_candidatos.csv (a baseline fica de fora), ou o indicado em
ESTAGIO1 abaixo.
"""
from collections import Counter
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier

import modelagem_comum as mc

ETAPA = "estagio2"
ESTAGIO1 = None  # None = melhor F1 do comparativo do 05
MINIMO_SMOTE = 5000  # SMOTE leva cada classe abaixo disso ate esse numero de fluxos


def random_forest(**kw):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=mc.SEMENTE, **kw)


def alvo_smote(y):
    return {c: MINIMO_SMOTE for c, n in Counter(y).items() if n < MINIMO_SMOTE}


# Secoes 11 e 12 de ../docs/DECISOES-E-BIBLIOGRAFIA.md
CANDIDATOS = {
    "rf_sem_balanceamento": random_forest(),
    "rf_class_weight": random_forest(class_weight="balanced"),
    "rf_smote": Pipeline([
        ("smote", SMOTE(sampling_strategy=alvo_smote, k_neighbors=5, random_state=mc.SEMENTE)),
        ("rf", random_forest()),
    ]),
}


def escolher_estagio1() -> str:
    if ESTAGIO1:
        return ESTAGIO1
    comp = pd.read_csv(mc.MET_DIR / "estagio1" / "comparativo_candidatos.csv", index_col=0)
    comp = comp.drop(index=[i for i in comp.index if i.startswith("baseline")])
    return comp["f1"].idxmax()


def validar_estagio2(modelo, X, y_multi, pred_estagio1, folds):
    """Devolve a previsao do estagio 2 para todos os fluxos de cada fold de validacao e as metricas por fold."""
    X_np = X.to_numpy()
    y_np = y_multi.to_numpy()
    ataque = y_np != 0
    pred2 = np.zeros(len(y_np), dtype=y_np.dtype)
    fold_de = np.zeros(len(y_np), dtype=np.int8)
    linhas = []

    for i, (tr, va) in enumerate(folds, start=1):
        tr_ataque = tr[ataque[tr]]
        m = clone(modelo)
        t0 = time.perf_counter()
        m.fit(X_np[tr_ataque], y_np[tr_ataque])
        tempo_treino = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred2[va] = m.predict(X_np[va])
        tempo_previsao = time.perf_counter() - t0
        fold_de[va] = i

        final = np.where(pred_estagio1[va] == 0, 0, pred2[va])
        va_ataque = va[ataque[va]]
        isolado = mc.metricas_multiclasse(y_np[va_ataque], pred2[va_ataque])
        completo = mc.metricas_multiclasse(y_np[va], final, classes=np.unique(y_np))
        metricas = {f"{k}_estagio2": v for k, v in isolado.items()}
        metricas.update({f"{k}_final": v for k, v in completo.items()})
        metricas.update({
            "fold": i,
            "tempo_treino_s": tempo_treino,
            "tempo_previsao_s": tempo_previsao,
            "tamanho_modelo_mb": mc.tamanho_serializado_mb(m),
        })
        linhas.append(metricas)
        print(f"  fold {i}/{len(folds)}: treino {tempo_treino:.0f}s, "
              f"f1_macro estagio2={isolado['f1_macro']:.4f}, final={completo['f1_macro']:.4f}")
    return pred2, fold_de, pd.DataFrame(linhas).set_index("fold")


if __name__ == "__main__":
    X, y_bin, y_multi = mc.carregar("train")
    nomes = mc.nomes_classes()
    folds = mc.folds_estratificados(y_multi)

    est1 = escolher_estagio1()
    print(f"Estagio 1 usado: {est1}")
    oof1 = pd.read_parquet(mc.PREV_DIR / "estagio1" / f"{est1}_oof.parquet")
    pred_estagio1 = oof1["y_pred"].to_numpy()
    assert (oof1["y_true"].to_numpy() == y_bin.to_numpy()).all(), "OOF do estagio 1 fora de ordem"

    (mc.MET_DIR / ETAPA).mkdir(parents=True, exist_ok=True)
    (mc.PREV_DIR / ETAPA).mkdir(parents=True, exist_ok=True)
    classes = np.unique(y_multi.to_numpy())

    finais = {}
    medias = {}
    for nome, modelo in CANDIDATOS.items():
        print(f"\n{nome}")
        pred2, fold_de, por_fold = validar_estagio2(modelo, X, y_multi, pred_estagio1, folds)
        resumo = mc.resumir_folds(por_fold)
        por_fold.to_csv(mc.MET_DIR / ETAPA / f"{nome}_por_fold.csv")
        resumo.to_csv(mc.MET_DIR / ETAPA / f"{nome}_resumo.csv")

        final = np.where(pred_estagio1 == 0, 0, pred2)
        pd.DataFrame({"fold": fold_de, "y_true": y_multi.to_numpy(), "pred_estagio2": pred2,
                      "y_pred_final": final}).to_parquet(mc.PREV_DIR / ETAPA / f"{nome}_oof.parquet", index=False)

        # por classe sobre as previsoes fora do fold reunidas (secao 10)
        mc.metricas_por_classe(y_multi.to_numpy(), final, classes, nomes).to_csv(
            mc.MET_DIR / ETAPA / f"{nome}_por_classe_final.csv")

        finais[nome] = final
        medias[nome] = resumo["media"]
        print(resumo.loc[["f1_macro_estagio2", "f1_macro_final", "acuracia_final", "tempo_treino_s"], :].round(4).to_string())

    comparativo = pd.DataFrame(medias).T.sort_values("f1_macro_final", ascending=False)
    comparativo.to_csv(mc.MET_DIR / ETAPA / "comparativo_candidatos.csv")
    print("\nComparativo (media dos folds, ordenado pelo F1 macro da previsao final):")
    print(comparativo[["f1_macro_estagio2", "f1_macro_final", "acuracia_final", "tempo_treino_s"]].round(4).to_string())

    pares = mc.mcnemar_pares(y_multi, finais)
    pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_pares.csv", index=False)
    print("\nMcNemar entre pares (previsao final em 15 classes):")
    print(pares.to_string(index=False))
