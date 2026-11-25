# Detecção de intrusões em rede no CIC-IDS2017

Classificação de tráfego de rede como benigno ou ataque, e identificação do
tipo de ataque, sobre a versão corrigida do CIC-IDS2017 (Engelen, Rimmer &
Joosen, 2021). O pipeline é hierárquico, em dois estágios: um classificador
binário detecta o ataque e um segundo classificador, treinado só nos fluxos
de ataque, identifica qual dos 14 tipos ele é. Um classificador multiclasse
direto, em estágio único, serve de referência. A validação usa SHAP para
checar se o modelo aprendeu padrão de ataque ou artefato do laboratório.

Projeto da disciplina PTC3567 - Ciência dos Dados em Automação e Engenharia,
Escola Politécnica da Universidade de São Paulo, 2º semestre de 2026, com os
professores Pedro Luiz Pizzigatti Corrêa e Felipe Miguel Pait.

**Autores:** Bianca Repke Teixeira, Eduardo Maravieski, Gabriel Cesar de
Almeida Lins e Pedro de Lemos Barros Reis.

## Resultados principais

Conjunto de teste (20% estratificado, avaliado uma única vez):

| | Resultado |
|---|---|
| Estágio 1 (binário, Random Forest com undersampling) | F1 do ataque 0,987; revocação 0,9995; AUC 0,9998 |
| Hierárquico, 15 classes | acurácia 0,995; F1 macro 0,921 |
| Multiclasse direto, 15 classes | acurácia 0,995; F1 macro 0,942 |
| McNemar hierárquico × direto | p = 0,090 (diferença não significativa) |

O gargalo está nas classes raras (XSS, SQL Injection, Infiltration, Web
Attack Brute Force), que têm de 2 a 30 fluxos no teste. A discussão completa,
com as hipóteses, as limitações e a comparação com PCA, está no
[relatório final](docs/relatorio-final.pdf), nos
[slides da apresentação](docs/slides-apresentacao.pdf) e em
[`docs/DECISOES-E-BIBLIOGRAFIA.md`](docs/DECISOES-E-BIBLIOGRAFIA.md).

## Estrutura

```
docs/
  relatorio-final.pdf               relatório final do projeto (25/11/2026)
  slides-apresentacao.pdf           apresentação do projeto (18/11/2026)
  DECISOES-E-BIBLIOGRAFIA.md        cada decisão do pipeline, o motivo e a referência
  PROPOSTA-RAG-EXPLICACAO-ALERTAS.md  extensão proposta, não implementada
scripts/
  01_carregar_e_limpar.py           carga dos CSVs, remoção de identificadores, inf/NaN e duplicatas
  02_rotulagem.py                   rótulos binário e multiclasse
  03_eda.ipynb / 03_eda_source.py   análise exploratória (notebook e .py equivalente)
  04_selecao_e_split.py             seleção por correlação, split 80/20, padronização
  05_estagio1_binario.py            estágio 1: candidatos e balanceamento, K-fold
  06_estagio2_tipo_ataque.py        estágio 2: tipo de ataque, previsão final em 15 classes
  07_multiclasse_direto.py          referência multiclasse em estágio único
  08_avaliacao_final.py             avaliação única no teste
  09_validacao_xai.py               SHAP global e local, comparação com a EDA
  10_graficos_modelagem.py          figuras da modelagem
  11_ e 12_candidatos_extras_*.py   Extra Trees, XGBoost e kNN nos três pontos
  13_comparacao_pca.py              desempenho com e sem PCA
  modelagem_comum.py                carga, métricas, K-fold, McNemar, tempo e memória
resultados/
  figuras/                          gráficos da EDA e da modelagem
  metricas/                         CSVs de métricas por etapa e logs de execução
rodar_pipeline.sh                   roda os scripts 01 a 13 em ordem
```

As previsões fora do fold (`resultados/previsoes/`), os dados processados e
os modelos treinados não estão no repositório por tamanho; o pipeline os
regenera.

## Como reproduzir

1. **Ambiente:** Python 3.12.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Dados:** baixe os CSVs da versão corrigida do CIC-IDS2017 na página dos
   autores, <https://intrusion-detection.distrinet-research.be/WTMC2021/>, e
   coloque os cinco arquivos (`Monday-WorkingHours.csv` a
   `Friday-WorkingHours.csv`) em `dados/raw/improved_cicids2017/`.

3. **Execução:**

   ```bash
   ./rodar_pipeline.sh
   ```

   Os scripts também rodam um a um, a partir de `scripts/`, na ordem
   numérica. Os scripts 11 a 13 são comparações extras e usam as previsões
   fora do fold dos scripts 05 a 07.

Tempo de referência, numa máquina de 8 núcleos e 15 GB de RAM, sem GPU:
cerca de 7 minutos por configuração no estágio 1 (5 folds). Os tempos de
treino e previsão de cada modelo estão nos CSVs `*_resumo.csv` de
`resultados/metricas/`.

## Licença

Código sob a [licença MIT](LICENSE). Documentos, figuras e resultados
(`docs/`, `resultados/`) sob a licença
[Creative Commons Atribuição 4.0 Internacional (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.pt_BR).

O dataset não é redistribuído aqui; ele segue os termos dos seus autores
(Canadian Institute for Cybersecurity e Engelen, Rimmer & Joosen).
