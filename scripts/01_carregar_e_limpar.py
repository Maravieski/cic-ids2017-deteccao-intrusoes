"""
Carga e limpeza do CICIDS2017 (versao corrigida - Engelen, Rimmer & Joosen, 2021).

Decisoes tomadas aqui e a justificativa/bibliografia de cada uma estao
documentadas em ../docs/DECISOES-E-BIBLIOGRAFIA.md. Resumo:

- Remocao de Flow ID, Src IP, Src Port, Dst IP, Timestamp: reproduz a limpeza
  do proprio codigo de referencia dos autores da versao corrigida do dataset
  (MakeDataNumpyFriendly.py, repo GintsEngelen/WTMC2021-Code), que evita que o
  modelo aprenda a topologia do laboratorio (IPs fixos de vitima/atacante) em
  vez do padrao de trafego -- problema de "shortcut learning" documentado no
  paper (Engelen et al., 2021).
- Flow Bytes/s e Flow Packets/s podem ser +-inf quando Flow Duration = 0
  (divisao por zero) -- inf vira NaN e as linhas com NaN sao descartadas
  (proporcao desprezivel frente aos ~2,1M fluxos).
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "dados" / "raw" / "improved_cicids2017"
OUT_DIR = Path(__file__).parent.parent / "dados" / "processado"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLUNAS_IDENTIFICADORAS = ["Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp"]

ARQUIVOS = [
    "Monday-WorkingHours.csv",
    "Tuesday-WorkingHours.csv",
    "Wednesday-WorkingHours.csv",
    "Thursday-WorkingHours.csv",
    "Friday-WorkingHours.csv",
]


def carregar() -> pd.DataFrame:
    partes = []
    for nome in ARQUIVOS:
        caminho = RAW_DIR / nome
        print(f"Carregando {nome}...")
        df = pd.read_csv(caminho, low_memory=False)
        df["Dia"] = nome.split("-")[0]
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    print(f"Total combinado: {df.shape[0]:,} fluxos, {df.shape[1]} colunas")
    return df


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    antes = len(df)

    df = df.drop(columns=[c for c in COLUNAS_IDENTIFICADORAS if c in df.columns])

    colunas_numericas = df.select_dtypes(include=[np.number]).columns
    df[colunas_numericas] = df[colunas_numericas].replace([np.inf, -np.inf], np.nan)

    linhas_com_nan = df[colunas_numericas].isna().any(axis=1).sum()
    print(f"Linhas com inf/NaN removidas: {linhas_com_nan:,} ({linhas_com_nan/antes:.4%})")
    df = df.dropna(subset=colunas_numericas)

    duplicadas = df.duplicated().sum()
    print(f"Linhas duplicadas removidas: {duplicadas:,} ({duplicadas/antes:.4%})")
    df = df.drop_duplicates()

    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

    print(f"Total apos limpeza: {len(df):,} fluxos ({len(df)/antes:.2%} do original combinado)")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = carregar()
    df = limpar(df)

    print("\nDistribuicao de rotulos (Label):")
    print(df["Label"].value_counts())

    destino = OUT_DIR / "dataset_limpo.parquet"
    df.to_parquet(destino, index=False)
    print(f"\nSalvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
