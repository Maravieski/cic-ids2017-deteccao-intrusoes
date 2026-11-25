"""
Funcoes compartilhadas pelos scripts de modelagem (05 em diante).

Implementa o protocolo de avaliacao da secao 10 de ../docs/DECISOES-E-BIBLIOGRAFIA.md:
- K-fold estratificado (K=5) so no conjunto de treino. Os folds sao
  estratificados pelo rotulo multiclasse mesmo quando o alvo e binario (mesmo
  criterio do split da secao 8, garante as classes raras em todo fold) e sao
  sempre os mesmos (semente fixa). Assim todos os candidatos de todos os
  scripts sao avaliados nas mesmas particoes, o que permite comparar previsoes
  fluxo a fluxo (McNemar) e encadear os dois estagios do pipeline hierarquico
  (secao 9): o 06 usa as previsoes fora do fold do estagio 1 salvas pelo 05.
- Balanceamento entra como etapa de um Pipeline do imbalanced-learn, que so
  reamostra no fit -- ou seja, so na parte de treino de cada fold, sem vazar
  amostra sintetica para a validacao.
- Metricas globais por fold (media e desvio padrao, slide "Resumo" de Analise
  de Desempenho de Modelos); metricas por classe sobre as previsoes fora do
  fold reunidas (cada fluxo do treino recebe exatamente uma previsao), porque
  as classes mais raras tem 1 ou 2 fluxos por fold de validacao.
- Tempo de treino/previsao e tamanho do modelo serializado em cada fold
  (o slide lista tempo de aprendizado e requisitos de armazenamento entre os
  aspectos de avaliacao).
- McNemar (Dietterich, 1998) com correcao de continuidade, ou binomial exato
  quando ha menos de 25 discordancias.
"""
from itertools import combinations
from pathlib import Path
import io
import json
import time

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).parent.parent
DADOS_DIR = BASE_DIR / "dados" / "processado"
MET_DIR = BASE_DIR / "resultados" / "metricas"
PREV_DIR = BASE_DIR / "resultados" / "previsoes"

SEMENTE = 42
K_FOLDS = 5


def carregar(conjunto: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """conjunto = "train" ou "test". X ja vem padronizado pelo 04; float32 para caber folgado na RAM."""
    X = pd.read_parquet(DADOS_DIR / f"X_{conjunto}.parquet").astype(np.float32)
    y_bin = pd.read_parquet(DADOS_DIR / f"y_bin_{conjunto}.parquet").iloc[:, 0]
    y_multi = pd.read_parquet(DADOS_DIR / f"y_multi_{conjunto}.parquet").iloc[:, 0]
    return X, y_bin, y_multi


def nomes_classes() -> dict[int, str]:
    with open(DADOS_DIR / "mapa_multiclasse.json") as f:
        mapa = json.load(f)
    return {codigo: nome for nome, codigo in mapa.items()}


def folds_estratificados(y_multi: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEMENTE)
    return list(skf.split(np.zeros(len(y_multi)), y_multi))


def tamanho_serializado_mb(modelo) -> float:
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    return buffer.tell() / 1e6


def validacao_cruzada(modelo, X: pd.DataFrame, y: pd.Series, folds, calcular_metricas):
    """
    Treina um clone do modelo em cada fold. Devolve:
    - pred: previsao fora do fold de cada fluxo;
    - score: probabilidade fora do fold por classe (n_fluxos x n_classes), ou None;
    - fold_de: em qual fold cada fluxo foi validado;
    - por_fold: DataFrame com as metricas de cada fold, mais tempo e tamanho.
    """
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    classes = np.unique(y_np)
    pred = np.empty(len(y_np), dtype=y_np.dtype)
    score = None
    fold_de = np.empty(len(y_np), dtype=np.int8)
    linhas = []

    for i, (tr, va) in enumerate(folds, start=1):
        m = clone(modelo)
        t0 = time.perf_counter()
        m.fit(X_np[tr], y_np[tr])
        tempo_treino = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred[va] = m.predict(X_np[va])
        proba = m.predict_proba(X_np[va]) if hasattr(m, "predict_proba") else None
        tempo_previsao = time.perf_counter() - t0

        if proba is not None:
            if score is None:
                score = np.zeros((len(y_np), len(classes)), dtype=np.float32)
            # classes_ do fold pode ser subconjunto das classes globais
            score[np.ix_(va, np.searchsorted(classes, m.classes_))] = proba
        fold_de[va] = i

        score_fold = score[va] if score is not None else None
        metricas = calcular_metricas(y_np[va], pred[va], score_fold)
        metricas.update({
            "fold": i,
            "tempo_treino_s": tempo_treino,
            "tempo_previsao_s": tempo_previsao,
            "tamanho_modelo_mb": tamanho_serializado_mb(m),
        })
        linhas.append(metricas)
        print(f"  fold {i}/{len(folds)}: treino {tempo_treino:.0f}s, "
              + ", ".join(f"{k}={v:.4f}" for k, v in metricas.items() if k in ("f1", "f1_macro", "revocacao", "especificidade")))

    return pred, score, fold_de, pd.DataFrame(linhas).set_index("fold")


def resumir_folds(por_fold: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"media": por_fold.mean(), "desvio_padrao": por_fold.std(ddof=1)})


def metricas_binarias(y_true, y_pred, score=None) -> dict:
    """Classe positiva = ataque (1). score: matriz de probabilidade (coluna 1 = ataque) ou None."""
    vn, fp, fn, vp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    especificidade = vn / (vn + fp) if vn + fp else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    auc = roc_auc_score(y_true, score[:, -1]) if score is not None else np.nan
    return {
        "acuracia": (vp + vn) / len(y_true),
        "precisao": precisao,
        "revocacao": revocacao,
        "especificidade": especificidade,
        "f1": f1,
        "fnr": 1 - revocacao,
        "tss": revocacao + especificidade - 1,
        "auc": auc,
        "vp": vp, "fn": fn, "fp": fp, "vn": vn,
    }


def metricas_por_classe(y_true, y_pred, classes, nomes: dict[int, str]) -> pd.DataFrame:
    """Cada classe contra as demais (um-contra-todos), a partir da matriz de confusao multiclasse."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    vp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - vp
    fp = cm.sum(axis=0) - vp
    vn = cm.sum() - vp - fn - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        precisao = np.nan_to_num(vp / (vp + fp))
        revocacao = np.nan_to_num(vp / (vp + fn))
        especificidade = np.nan_to_num(vn / (vn + fp))
        f1 = np.nan_to_num(2 * precisao * revocacao / (precisao + revocacao))
    return pd.DataFrame({
        "classe": classes,
        "nome": [nomes.get(int(c), str(c)) for c in classes],
        "suporte": (vp + fn).astype(int),
        "precisao": precisao,
        "revocacao": revocacao,
        "especificidade": especificidade,
        "f1": f1,
        "fnr": 1 - revocacao,
        "tss": revocacao + especificidade - 1,
    }).set_index("classe")


def metricas_multiclasse(y_true, y_pred, score=None, classes=None) -> dict:
    """Medias macro (cada classe pesa igual, secao 10). score nao e usado: AUC so no binario."""
    classes = np.unique(y_true) if classes is None else classes
    por_classe = metricas_por_classe(y_true, y_pred, classes, {})
    return {
        "acuracia": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "f1_macro": por_classe["f1"].mean(),
        "precisao_macro": por_classe["precisao"].mean(),
        "revocacao_macro": por_classe["revocacao"].mean(),
        "especificidade_macro": por_classe["especificidade"].mean(),
        "tss_macro": por_classe["tss"].mean(),
    }


def mcnemar(y_true, pred_a, pred_b) -> dict:
    """
    b = fluxos que so A acerta, c = fluxos que so B acerta. Com 1,2M fluxos quase
    qualquer diferenca da significativa, por isso a diferenca de acuracia (b - c)/n
    vai junto do p-valor como tamanho do efeito.
    """
    y_true = np.asarray(y_true)
    acerto_a = np.asarray(pred_a) == y_true
    acerto_b = np.asarray(pred_b) == y_true
    b = int(np.sum(acerto_a & ~acerto_b))
    c = int(np.sum(~acerto_a & acerto_b))
    if b + c == 0:
        estatistica, p_valor = 0.0, 1.0
    elif b + c < 25:
        estatistica, p_valor = np.nan, stats.binomtest(min(b, c), b + c, 0.5).pvalue
    else:
        estatistica = (abs(b - c) - 1) ** 2 / (b + c)
        p_valor = stats.chi2.sf(estatistica, df=1)
    return {
        "so_a_acerta": b,
        "so_b_acerta": c,
        "estatistica": estatistica,
        "p_valor": p_valor,
        "diferenca_acuracia": (b - c) / len(y_true),
    }


def mcnemar_pares(y_true, previsoes: dict[str, np.ndarray]) -> pd.DataFrame:
    linhas = []
    for a, b in combinations(previsoes, 2):
        linhas.append({"modelo_a": a, "modelo_b": b, **mcnemar(y_true, previsoes[a], previsoes[b])})
    return pd.DataFrame(linhas)


def salvar_validacao(etapa: str, nome: str, por_fold: pd.DataFrame, resumo: pd.DataFrame,
                     y_true, pred, score, fold_de) -> None:
    (MET_DIR / etapa).mkdir(parents=True, exist_ok=True)
    (PREV_DIR / etapa).mkdir(parents=True, exist_ok=True)
    por_fold.to_csv(MET_DIR / etapa / f"{nome}_por_fold.csv")
    resumo.to_csv(MET_DIR / etapa / f"{nome}_resumo.csv")

    oof = pd.DataFrame({"fold": fold_de, "y_true": np.asarray(y_true), "y_pred": pred})
    if score is not None:
        # mesma ordem de colunas usada em validacao_cruzada (classes ordenadas)
        for j, classe in enumerate(np.unique(np.asarray(y_true))):
            oof[f"score_{classe}"] = score[:, j]
    oof.to_parquet(PREV_DIR / etapa / f"{nome}_oof.parquet", index=False)
