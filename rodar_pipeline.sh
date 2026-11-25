#!/bin/bash
# Roda o pipeline completo, na ordem. Um script falhando interrompe a cadeia.
# Pre-requisito: CSVs do dataset em dados/raw/improved_cicids2017/ (ver README).
# Logs de cada etapa em resultados/metricas/log_cadeia/.
set -e
cd "$(dirname "$0")/scripts"
LOG=../resultados/metricas/log_cadeia
mkdir -p "$LOG"
PY=${PYTHON:-python}
for s in 01_carregar_e_limpar 02_rotulagem 03_eda_source 04_selecao_e_split \
         05_estagio1_binario 06_estagio2_tipo_ataque 07_multiclasse_direto \
         08_avaliacao_final 09_validacao_xai 10_graficos_modelagem \
         11_candidatos_extras_estagio1 12_candidatos_extras_estagio2_e_direto \
         13_comparacao_pca; do
  echo "$(date +%T) inicio $s" | tee -a "$LOG/progresso.txt"
  "$PY" -u "$s.py" > "$LOG/$s.txt" 2>&1 || { echo "$(date +%T) FALHOU $s" | tee -a "$LOG/progresso.txt"; exit 1; }
  echo "$(date +%T) fim $s" | tee -a "$LOG/progresso.txt"
done
