"""
Rotulagem final: funde variantes "- Attempted" em BENIGN e cria os alvos
binario e multiclasse.

Justificativa (ver DECISOES-E-BIBLIOGRAFIA.md): os proprios autores da versao
corrigida do dataset (Engelen, Rimmer & Joosen, 2021; script
labelling_CSV_flows.py / MakeDataNumpyFriendly.py do repo GintsEngelen/
WTMC2021-Code) tratam todo fluxo "X - Attempted" como BENIGN em seus
experimentos de benchmark, com o argumento de que a tentativa nao chegou a
entregar payload -- comportamento de rede mais proximo do trafego benigno do
que do ataque completo. Reproduzimos essa escolha para manter compatibilidade
com a literatura de referencia do proprio dataset.

O mapeamento numerico multiclasse reproduz o dicionario de
MakeDataNumpyFriendly.py (0=BENIGN, 1..14 = categorias de ataque).
"""
from pathlib import Path
import json
import pandas as pd

IN_PATH = Path(__file__).parent.parent / "dados" / "processado" / "dataset_limpo.parquet"
OUT_DIR = Path(__file__).parent.parent / "dados" / "processado"

MAPA_MULTICLASSE = {
    "BENIGN": 0,
    "FTP-Patator": 1,
    "SSH-Patator": 2,
    "DoS GoldenEye": 3,
    "DoS Hulk": 4,
    "DoS Slowhttptest": 5,
    "DoS slowloris": 6,
    "Heartbleed": 7,
    "Web Attack - Brute Force": 8,
    "Web Attack - XSS": 9,
    "Web Attack - Sql Injection": 10,
    "Infiltration": 11,
    "Bot": 12,
    "PortScan": 13,
    "DDoS": 14,
}

RENOMEIA_PARA_BENIGN = [f"{k} - Attempted" for k in MAPA_MULTICLASSE if k != "BENIGN"]


def rotular(df: pd.DataFrame) -> pd.DataFrame:
    df["Label"] = df["Label"].replace({rotulo: "BENIGN" for rotulo in RENOMEIA_PARA_BENIGN})

    nao_mapeados = set(df["Label"].unique()) - set(MAPA_MULTICLASSE)
    if nao_mapeados:
        raise ValueError(f"Rotulos nao mapeados encontrados: {nao_mapeados}")

    df["Label_multiclasse"] = df["Label"].map(MAPA_MULTICLASSE).astype("int16")
    df["Label_binario"] = (df["Label"] != "BENIGN").astype("int8")
    return df


if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    df = rotular(df)

    print("Distribuicao final (multiclasse):")
    print(df["Label"].value_counts())
    print("\nDistribuicao binaria:")
    print(df["Label_binario"].value_counts(normalize=True))

    destino = OUT_DIR / "dataset_rotulado.parquet"
    df.to_parquet(destino, index=False)

    with open(OUT_DIR / "mapa_multiclasse.json", "w") as f:
        json.dump(MAPA_MULTICLASSE, f, ensure_ascii=False, indent=2)

    print(f"\nSalvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
