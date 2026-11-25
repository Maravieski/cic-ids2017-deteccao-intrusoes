"""
Selecao de atributos e divisao treino/teste.

Decisoes (ver DECISOES-E-BIBLIOGRAFIA.md):
0. Protocol e Dst Port sao numericos na coluna mas qualitativos nominais no
   significado (Aula 2, "armadilha classica" -- mesmo caso do exemplo Id/CEP
   do slide: media/correlacao de Pearson nao fazem sentido neles). Por isso
   ficam de fora dos filtros de variancia/correlacao (pensados pra atributo
   quantitativo) e voltam intactos pro conjunto final de atributos -- sao
   uteis pro classificador (ataques miram portas/protocolos especificos), so
   nao passam pelo mesmo teste estatistico dos quantitativos.
1. Remove colunas de variancia zero (nao carregam informacao).
2. Remove colunas com correlacao de Pearson > 0.95 em modulo com outra ja
   mantida -- reducao de redundancia/dimensionalidade, no mesmo espirito da
   etapa de "feature extraction" usada por Talukder et al. (2024) para lidar
   com a alta dimensionalidade do CICIDS2017 (naquele trabalho via PCA; aqui
   optou-se por um filtro de correlacao, mais simples e mais facil de
   interpretar/reportar no relatorio da disciplina).
3. Divisao estratificada 80/20 pelo rotulo multiclasse, para garantir que
   classes raras (ex.: Heartbleed, 11 fluxos) apareçam em treino e teste --
   pratica padrao para dados desbalanceados (Kelleher & Tierney, 2018,
   "Data Science", cap. sobre avaliacao de modelos).
4. Padronizacao (StandardScaler) ajustada SOMENTE no treino e aplicada ao
   teste, para evitar vazamento de informacao (data leakage) do conjunto de
   teste para o pipeline de treino -- reproduz a "normalizacao por
   padronizacao" da Aula 4, escolhida em vez da normalizacao por reescala por
   lidar melhor com outlier.
5. Comparacao quantitativa com PCA (a pedido do professor, 23/09/2026): sobre
   os atributos quantitativos ja padronizados, quanto da variancia um PCA
   explicaria -- so pra checar com numero real se o filtro de correlacao do
   passo 2 (que optou por preservar atributos interpretaveis em vez de
   componentes) deixou muito ganho de reducao na mesa. Nao afeta X_train/
   X_test/scaler nem o conjunto final de atributos, e so um relatorio a mais.
"""
from pathlib import Path
import json
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IN_PATH = Path(__file__).parent.parent / "dados" / "processado" / "dataset_rotulado.parquet"
OUT_DIR = Path(__file__).parent.parent / "dados" / "processado"
MET_DIR = Path(__file__).parent.parent / "resultados" / "metricas"
FIG_DIR = Path(__file__).parent.parent / "resultados" / "figuras"

COLUNAS_NAO_ATRIBUTO = ["Label", "Label_multiclasse", "Label_binario", "Dia"]
ATRIBUTOS_QUALITATIVOS_NOMINAIS = ["Protocol", "Dst Port"]
LIMIAR_CORRELACAO = 0.95
AMOSTRA_PCA = 300_000


def remover_variancia_zero(df: pd.DataFrame, colunas: list[str]) -> list[str]:
    variancias = df[colunas].var()
    mantidas = variancias[variancias > 0].index.tolist()
    removidas = sorted(set(colunas) - set(mantidas))
    print(f"Removidas por variancia zero ({len(removidas)}): {removidas}")
    return mantidas


def remover_alta_correlacao(df: pd.DataFrame, colunas: list[str], limiar: float) -> list[str]:
    amostra = df[colunas].sample(n=min(300_000, len(df)), random_state=42)
    corr = amostra.corr(numeric_only=True).abs()
    superior = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    para_remover = [col for col in superior.columns if any(superior[col] > limiar)]
    mantidas = [c for c in colunas if c not in para_remover]
    print(f"Removidas por alta correlacao (>{limiar}) ({len(para_remover)}): {para_remover}")
    return mantidas


def comparar_com_pca(X_train_esc: pd.DataFrame, colunas_quantitativas: list[str]) -> None:
    amostra = X_train_esc[colunas_quantitativas].sample(n=min(AMOSTRA_PCA, len(X_train_esc)), random_state=42)

    pca = PCA(random_state=42)
    pca.fit(amostra)
    var_exp = pca.explained_variance_ratio_
    var_cum = np.cumsum(var_exp)
    n_90 = int(np.searchsorted(var_cum, 0.90) + 1)
    n_95 = int(np.searchsorted(var_cum, 0.95) + 1)

    resumo = pd.DataFrame({
        "componente": np.arange(1, len(var_exp) + 1),
        "variancia_explicada": var_exp,
        "variancia_explicada_cumulativa": var_cum,
    })
    resumo.to_csv(MET_DIR / "pca_variancia_explicada.csv", index=False)
    print(f"\nPCA (comparacao, {len(colunas_quantitativas)} atributos quantitativos, amostra {len(amostra):,}):")
    print(f"  Componentes para 90% da variancia: {n_90}")
    print(f"  Componentes para 95% da variancia: {n_95}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(resumo["componente"], resumo["variancia_explicada"], color="#4C72B0", alpha=0.7, label="Variância explicada (individual)")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Variância explicada (individual)", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    ax2 = ax1.twinx()
    ax2.plot(resumo["componente"], resumo["variancia_explicada_cumulativa"], color="#C44E52", marker="o", markersize=3, label="Variância explicada (cumulativa)")
    ax2.axhline(0.90, color="grey", linestyle="--", linewidth=1)
    ax2.axhline(0.95, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("Variância explicada (cumulativa)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, 1.02)

    ax1.set_title(f"PCA sobre os {len(colunas_quantitativas)} atributos quantitativos padronizados (amostra de {len(amostra)//1000}k fluxos)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_variancia_explicada.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    colunas_quantitativas = [
        c for c in df.select_dtypes(include="number").columns
        if c not in COLUNAS_NAO_ATRIBUTO and c not in ATRIBUTOS_QUALITATIVOS_NOMINAIS
    ]

    colunas_quantitativas = remover_variancia_zero(df, colunas_quantitativas)
    colunas_quantitativas = remover_alta_correlacao(df, colunas_quantitativas, LIMIAR_CORRELACAO)

    colunas_atributo = colunas_quantitativas + ATRIBUTOS_QUALITATIVOS_NOMINAIS
    print(f"\nAtributos quantitativos apos filtro: {len(colunas_quantitativas)}")
    print(f"Atributos qualitativos nominais mantidos sem filtro: {ATRIBUTOS_QUALITATIVOS_NOMINAIS}")
    print(f"Atributos finais: {len(colunas_atributo)}")

    X = df[colunas_atributo]
    y_bin = df["Label_binario"]
    y_multi = df["Label_multiclasse"]

    X_train, X_test, y_bin_train, y_bin_test, y_multi_train, y_multi_test = train_test_split(
        X, y_bin, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )

    scaler = StandardScaler()
    X_train_esc = pd.DataFrame(scaler.fit_transform(X_train), columns=colunas_atributo, index=X_train.index)
    X_test_esc = pd.DataFrame(scaler.transform(X_test), columns=colunas_atributo, index=X_test.index)

    X_train_esc.to_parquet(OUT_DIR / "X_train.parquet")
    X_test_esc.to_parquet(OUT_DIR / "X_test.parquet")
    y_bin_train.to_frame().to_parquet(OUT_DIR / "y_bin_train.parquet")
    y_bin_test.to_frame().to_parquet(OUT_DIR / "y_bin_test.parquet")
    y_multi_train.to_frame().to_parquet(OUT_DIR / "y_multi_train.parquet")
    y_multi_test.to_frame().to_parquet(OUT_DIR / "y_multi_test.parquet")

    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    with open(OUT_DIR / "colunas_atributo.json", "w") as f:
        json.dump(colunas_atributo, f, ensure_ascii=False, indent=2)

    contagem_treino = y_multi_train.value_counts().sort_index()
    contagem_teste = y_multi_test.value_counts().sort_index()
    resumo_split = pd.DataFrame({"treino": contagem_treino, "teste": contagem_teste}).fillna(0).astype(int)
    resumo_split.to_csv(MET_DIR / "resumo_split.csv")
    print("\nContagem por classe (treino/teste):")
    print(resumo_split)

    print(f"\nTreino: {X_train_esc.shape}, Teste: {X_test_esc.shape}")

    comparar_com_pca(X_train_esc, colunas_quantitativas)
