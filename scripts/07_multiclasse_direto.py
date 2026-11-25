"""
Referencia da secao 9 de ../docs/DECISOES-E-BIBLIOGRAFIA.md: classificador multiclasse
direto (15 classes, estagio unico), com o mesmo algoritmo do pipeline hierarquico.
Serve para medir o ganho do hierarquico, e nao presumi-lo.

Mesmo protocolo e mesmos folds dos scripts 05 e 06 (secao 10). As previsoes fora
do fold ficam em ../resultados/previsoes/multiclasse_direto/ e entram no McNemar
contra a previsao final do 06 (comparativo feito no 08).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import modelagem_comum as mc

ETAPA = "multiclasse_direto"

CANDIDATOS = {
    "rf_sem_balanceamento": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=mc.SEMENTE),
    "rf_class_weight": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=mc.SEMENTE,
                                              class_weight="balanced"),
}

if __name__ == "__main__":
    X, y_bin, y_multi = mc.carregar("train")
    nomes = mc.nomes_classes()
    folds = mc.folds_estratificados(y_multi)
    classes = np.unique(y_multi.to_numpy())
    print(f"Treino: {X.shape[0]:,} fluxos, {X.shape[1]} atributos, {len(folds)} folds")

    medias = {}
    previsoes = {}
    for nome, modelo in CANDIDATOS.items():
        print(f"\n{nome}")
        pred, score, fold_de, por_fold = mc.validacao_cruzada(
            modelo, X, y_multi, folds, lambda y, p, s: mc.metricas_multiclasse(y, p, s, classes=classes))
        resumo = mc.resumir_folds(por_fold)
        mc.salvar_validacao(ETAPA, nome, por_fold, resumo, y_multi, pred, score, fold_de)
        mc.metricas_por_classe(y_multi.to_numpy(), pred, classes, nomes).to_csv(
            mc.MET_DIR / ETAPA / f"{nome}_por_classe.csv")
        previsoes[nome] = pred
        medias[nome] = resumo["media"]
        print(resumo.round(4).to_string())

    comparativo = pd.DataFrame(medias).T.sort_values("f1_macro", ascending=False)
    comparativo.to_csv(mc.MET_DIR / ETAPA / "comparativo_candidatos.csv")
    print("\nComparativo (media dos folds, ordenado pelo F1 macro):")
    print(comparativo[["f1_macro", "acuracia", "revocacao_macro", "tempo_treino_s"]].round(4).to_string())

    pares = mc.mcnemar_pares(y_multi, previsoes)
    pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_pares.csv", index=False)
    print("\nMcNemar entre pares:")
    print(pares.to_string(index=False))
