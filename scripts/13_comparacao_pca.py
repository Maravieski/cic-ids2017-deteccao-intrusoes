"""
Comparacao de desempenho com e sem PCA no estagio 1 (addendum da secao 7 de
../docs/DECISOES-E-BIBLIOGRAFIA.md). Em 23/09/2026 o 04 comparou o PCA so pela variancia
explicada (22 componentes para 90%, 27 para 95%, sobre os 49 atributos quantitativos).
Aqui a comparacao e pelo que o classificador acerta.

Candidatos, todos com o modelo escolhido para o estagio 1 (Random Forest de 100 arvores
com undersampling do BENIGN, script 05):
- rf_undersampling: os 51 atributos da selecao por correlacao (resultado ja salvo pelo 05,
  reaproveitado, nao e retreinado);
- pca27_undersampling: os 49 quantitativos trocados por 27 componentes (95% da variancia),
  Protocol e Dst Port passam sem mudanca, porque sao qualitativos (secao 3);
- pca22_undersampling: idem, com 22 componentes (90% da variancia).

O PCA fica dentro do Pipeline: e ajustado so na parte de treino de cada fold, antes do
undersampling (os componentes saem da distribuicao real do treino), sem ver a validacao.
Mesmo protocolo do 05 (K=5, mesmos folds, F1 do ataque, McNemar contra o rf_undersampling).
Resultados em resultados/metricas/estagio1_pca/. O teste nao e tocado aqui.
"""
import json

import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

import modelagem_comum as mc

ETAPA = "estagio1_pca"
S = mc.SEMENTE
QUALITATIVOS = ["Protocol", "Dst Port"]


def candidato_pca(n_componentes: int, colunas: list[str]) -> Pipeline:
    quant = [i for i, c in enumerate(colunas) if c not in QUALITATIVOS]
    qual = [i for i, c in enumerate(colunas) if c in QUALITATIVOS]
    return Pipeline([
        ("pca", ColumnTransformer([
            ("pca", PCA(n_components=n_componentes, random_state=S), quant),
            ("qualitativos", "passthrough", qual),
        ])),
        ("under", RandomUnderSampler(random_state=S)),
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=S)),
    ])


def deteccao_comparada(y_multi: pd.Series) -> pd.DataFrame:
    """Fluxos de cada tipo marcados como ataque (fora do fold), com e sem PCA."""
    nomes = mc.nomes_classes()
    tabela = pd.DataFrame({"suporte": y_multi.value_counts()})
    for nome, pasta in [("rf_undersampling", "estagio1"), ("pca27_undersampling", ETAPA), ("pca22_undersampling", ETAPA)]:
        pred = pd.read_parquet(mc.PREV_DIR / pasta / f"{nome}_oof.parquet")["y_pred"].to_numpy()
        tabela[f"detectados_{nome}"] = pd.Series(pred == 1, index=y_multi.index).groupby(y_multi).sum()
    tabela.insert(0, "nome", [nomes[c] for c in tabela.index])
    return tabela.sort_values("suporte")


if __name__ == "__main__":
    X, y_bin, y_multi = mc.carregar("train")
    folds = mc.folds_estratificados(y_multi)
    with open(mc.DADOS_DIR / "colunas_atributo.json") as f:
        colunas = json.load(f)
    assert list(X.columns) == colunas

    candidatos = {
        "pca27_undersampling": candidato_pca(27, colunas),
        "pca22_undersampling": candidato_pca(22, colunas),
    }
    previsoes = {"rf_undersampling": pd.read_parquet(mc.PREV_DIR / "estagio1" / "rf_undersampling_oof.parquet")["y_pred"].to_numpy()}
    medias = {"rf_undersampling": pd.read_csv(mc.MET_DIR / "estagio1" / "rf_undersampling_resumo.csv", index_col=0)["media"]}

    for nome, modelo in candidatos.items():
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
    print(comparativo[["f1", "acuracia", "precisao", "revocacao", "especificidade", "auc",
                       "fp", "fn", "tempo_treino_s", "tempo_previsao_s", "tamanho_modelo_mb"]].round(4).to_string())
    pares = mc.mcnemar_pares(y_bin, previsoes)
    pares.to_csv(mc.MET_DIR / ETAPA / "mcnemar_pares.csv", index=False)
    print("\nMcNemar:")
    print(pares.to_string(index=False))

    por_tipo = deteccao_comparada(y_multi)
    por_tipo.to_csv(mc.MET_DIR / ETAPA / "deteccao_por_tipo_comparada.csv")
    print("\nDeteccao por tipo (fora do fold):")
    print(por_tipo.to_string())
