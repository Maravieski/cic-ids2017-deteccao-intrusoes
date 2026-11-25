"""
Estagio 1 do pipeline hierarquico (secao 9 de ../docs/DECISOES-E-BIBLIOGRAFIA.md):
classificador binario BENIGN x ataque, treinado em todo o conjunto de treino.
Testa a H1 da proposta.

Protocolo de avaliacao da secao 10, implementado em modelagem_comum.py:
K-fold estratificado (K=5) so no treino, metrica principal F1 da classe ataque,
demais metricas da aula reportadas, McNemar entre os candidatos. O conjunto de
teste so entra no modelo final, depois da escolha.

Alem das metricas binarias, calcula a taxa de deteccao por tipo de ataque
(fracao de cada classe original prevista como ataque): mostra quais tipos o
detector deixa passar, que e o que limita o estagio 2 e o que a H2 discute.

Candidatos (11/11/2026): baseline da classe majoritaria (Aula 4, "Dados
desbalanceados": a acuracia de qualquer classificador precisa superar a de
atribuir tudo a classe majoritaria) e Random Forest com tres estrategias de
balanceamento (secoes 11 e 12).
"""
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

import modelagem_comum as mc

ETAPA = "estagio1"

def random_forest(**kw):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=mc.SEMENTE, **kw)


# Secoes 11 e 12 de ../docs/DECISOES-E-BIBLIOGRAFIA.md: Random Forest, tres estrategias de
# balanceamento no estagio 1. O undersampling fica num Pipeline para so atuar no fit.
CANDIDATOS = {
    "baseline_majoritaria": DummyClassifier(strategy="most_frequent"),
    "rf_sem_balanceamento": random_forest(),
    "rf_class_weight": random_forest(class_weight="balanced"),
    "rf_undersampling": Pipeline([
        ("under", RandomUnderSampler(random_state=mc.SEMENTE)),
        ("rf", random_forest()),
    ]),
}


def deteccao_por_tipo(y_multi: pd.Series, pred, nomes: dict[int, str]) -> pd.DataFrame:
    detectado = pd.Series(pred == 1, index=y_multi.index)
    tabela = detectado.groupby(y_multi).agg(["sum", "count"])
    tabela.columns = ["previstos_como_ataque", "suporte"]
    tabela["taxa_deteccao"] = tabela["previstos_como_ataque"] / tabela["suporte"]
    tabela.insert(0, "nome", [nomes[c] for c in tabela.index])
    return tabela


if __name__ == "__main__":
    X, y_bin, y_multi = mc.carregar("train")
    nomes = mc.nomes_classes()
    folds = mc.folds_estratificados(y_multi)
    print(f"Treino: {X.shape[0]:,} fluxos, {X.shape[1]} atributos, {len(folds)} folds")

    previsoes = {}
    medias = {}
    for nome, modelo in CANDIDATOS.items():
        print(f"\n{nome}")
        pred, score, fold_de, por_fold = mc.validacao_cruzada(modelo, X, y_bin, folds, mc.metricas_binarias)
        resumo = mc.resumir_folds(por_fold)
        mc.salvar_validacao(ETAPA, nome, por_fold, resumo, y_bin, pred, score, fold_de)

        por_tipo = deteccao_por_tipo(y_multi, pred, nomes)
        por_tipo.to_csv(mc.MET_DIR / ETAPA / f"{nome}_deteccao_por_tipo.csv")

        previsoes[nome] = pred
        medias[nome] = resumo["media"]
        print(resumo.round(4).to_string())

    comparativo = pd.DataFrame(medias).T.sort_values("f1", ascending=False)
    comparativo.to_csv(mc.MET_DIR / ETAPA / "comparativo_candidatos.csv")
    print("\nComparativo (media dos folds, ordenado pelo F1):")
    print(comparativo[["f1", "acuracia", "precisao", "revocacao", "especificidade", "fnr", "tss", "auc", "tempo_treino_s"]].round(4).to_string())

    if len(previsoes) > 1:
        pares = mc.mcnemar_pares(y_bin, previsoes)
        pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_pares.csv", index=False)
        print("\nMcNemar entre pares de candidatos (previsoes fora do fold):")
        print(pares.to_string(index=False))
