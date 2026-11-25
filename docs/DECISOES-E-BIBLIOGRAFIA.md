# Decisões metodológicas e bibliografia

Este arquivo documenta, na ordem de execução, cada decisão tomada nos scripts
de `scripts/`, com a referência que a embasa. O relatório final (`docs/`) é a
versão redigida; aqui fica o registro de cada escolha e do motivo dela.

**Hipóteses do projeto**, citadas ao longo do texto como H1 a H4:

- **H1:** modelos baseados em árvores em ensemble (Random Forest, Gradient
  Boosted Trees, Extra Trees) atingem alto desempenho na classificação binária
  (benigno × ataque) sobre o CIC-IDS2017, replicando o padrão já reportado na
  literatura (Faker & Dogdu, 2019; Talukder et al., 2024).
- **H2:** a classificação multiclasse (identificar qual das 14 categorias de
  ataque ocorreu) é mais difícil que a binária, com queda de desempenho
  concentrada nas categorias minoritárias (Heartbleed, SQL Injection,
  Infiltration).
- **H3:** técnicas de tratamento de desbalanceamento melhoram a detecção das
  categorias de ataque minoritárias em relação a um treino direto sobre o
  dataset bruto.
- **H4:** o volume do dataset impõe restrições de processamento que
  justificam avaliar abordagens de Big Data (ex.: Apache Spark) frente a um
  pipeline convencional de máquina única.

**Referências às aulas:** o projeto foi feito na disciplina PTC3567 - Ciência
dos Dados em Automação e Engenharia (Escola Politécnica da USP, 2º semestre de
2026). As menções a "Aula 1", "Aula 2", "Aula 4" e aos slides de "Análise de
Desempenho de Modelos" se referem aos slides dessa disciplina, que são material
interno e não fazem parte deste repositório. O conceito usado de cada aula é
descrito no próprio texto.

**Formato dos scripts:** `.py` por padrão (etapas de ETL, split e modelagem).
O 03 (EDA) é notebook (`.ipynb`, com o `.py` equivalente em formato
jupytext), porque é a etapa mais visual e exploratória.

## 1. Fonte dos dados: versão corrigida do CICIDS2017

**Decisão:** usar a versão corrigida/relabeled do CICIDS2017 publicada pelos
próprios pesquisadores que auditaram o dataset original, em vez do
`MachineLearningCVE` distribuído pelo CIC/UNB.

**Por quê:** a fonte oficial (unb.ca/cicresearch.ca) hoje só libera o dataset
via formulário de solicitação (gate para fins estatísticos deles), sem link
de download direto. Mais relevante que a questão de acesso: o dataset
original tem problemas documentados de rotulagem e construção de fluxos.

**Referência:** ENGELEN, Gints; RIMMER, Vera; JOOSEN, Wouter. *Troubleshooting
an Intrusion Detection Dataset: the CICIDS2017 Case Study*. IEEE Security and
Privacy Workshops (SPW) / WTMC 2021. Dataset e código:
<https://github.com/GintsEngelen/WTMC2021-Code>,
<https://intrusion-detection.distrinet-research.be/WTMC2021/>.

**Efeito colateral a registrar:** a proposta de projeto do grupo cita estatísticas do dataset original (2.830.540
fluxos, Heartbleed=11, SQL Injection=21, Infiltration=36). Com a versão
corrigida, após limpeza os números mudam ligeiramente - é esperado, pois a
correção remove duplicatas e fluxos malformados do dataset original. O tema,
a questão central e as hipóteses (H1-H4) da proposta continuam válidos e não
precisam ser reescritos.

## 2. Remoção de colunas identificadoras

**Decisão:** remover `Flow ID`, `Src IP`, `Src Port`, `Dst IP` e `Timestamp`
antes de qualquer análise (mantendo `Dst Port` e `Protocol`).

**Por quê:** reproduz a limpeza do próprio código de referência dos autores
da versão corrigida (`MakeDataNumpyFriendly.py`), que evita "shortcut
learning" - o modelo aprender a topologia fixa do laboratório (qual IP era o
atacante) em vez do padrão de tráfego. Também é uma aplicação direta da
**eliminação manual de atributos** ensinada na Aula 4: atributos
identificadores (o slide usa o exemplo de Id/RG num prontuário médico) não
contribuem para a análise e podem ser descartados por julgamento de domínio,
sem precisar de técnica estatística.

**Referência:** ENGELEN, RIMMER & JOOSEN (2021), já citado acima; e Aula 4 -
Seleção de Dados, seção "Eliminação manual de atributos".

## 3. Classificação de atributos: Protocol e Dst Port são qualitativos, não quantitativos

**Decisão:** tratar `Protocol` (3 valores: 6=TCP, 17=UDP, 0=outro) e
`Dst Port` como atributos **qualitativos nominais**, mesmo estando
codificados como número - excluídos do `describe()`, do IQR/outlier e da
matriz de correlação de Pearson (que só fazem sentido para quantitativo),
recebendo em vez disso uma tabela de frequência. No filtro de seleção de
atributos (item 6 abaixo), os dois ficam de fora do teste de correlação, mas
voltam intactos ao conjunto final de atributos - são relevantes para o
classificador (ataques miram portas e protocolos específicos), só não fazem
sentido no teste estatístico pensado para quantitativo.

**Por quê:** a Aula 2 identifica isso como a "armadilha clássica" da EDA -
atributo qualitativo às vezes é representado por número (o slide usa Id e CEP
como exemplo), e calcular a média desse número não tem significado ("a média
dos Ids"). `Protocol` é exatamente esse caso: é um código categórico (3
valores possíveis), não uma quantidade. Classificar corretamente o tipo de
cada atributo é, segundo o slide, "a primeira decisão da EDA", porque dela
dependem métrica, visualização e modelo escolhidos depois.

**Referência:** Aula 2 - Caracterização e Análise Exploratória dos Dados,
seção "Tipos de atributos".

## 4. Tratamento de valores infinitos/NaN e de duplicatas

**Decisão:** `Flow Bytes/s` e `Flow Packets/s` podem virar ±infinito quando
`Flow Duration = 0` (divisão por zero); esses valores viram `NaN` e a linha é
descartada (0,038% do total combinado). Linhas 100% duplicadas também são
descartadas (186.992 de 2.100.814, 8,90%).

**Por quê:** o problema de inf/NaN em fluxos com duração zero está
documentado desde o paper original do dataset. Já a remoção de duplicata
exata é a aplicação direta do conceito de **redundância de objeto** da Aula
4 - "linhas quase idênticas" (aqui, idênticas de verdade) não agregam
informação nova e podem enviesar estatística e treino.

**Referência:** SHARAFALDIN, Iman; LASHKARI, Arash Habibi; GHORBANI, Ali A.
*Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic
Characterization*. ICISSP 2018;
ENGELEN, RIMMER & JOOSEN (2021); Aula 4, seção "Dados inconsistentes e
redundantes".

## 5. Fusão de rótulos "- Attempted" em BENIGN

**Decisão:** todo fluxo rotulado como `<Ataque> - Attempted` (ex.: `FTP-Patator
- Attempted`) passa a ser tratado como `BENIGN`.

**Por quê:** reproduz a metodologia de benchmark dos autores da versão
corrigida - uma tentativa que não chegou a entregar payload é, em termos de
padrão de tráfego observável, mais próxima do tráfego benigno do que do
ataque completo. (Decisão específica da fonte de dados, não uma técnica geral
de aula.)

**Referência:** ENGELEN, RIMMER & JOOSEN (2021), comentário em
`labelling_CSV_flows.py` / `MakeDataNumpyFriendly.py`.

## 6. Estatística descritiva: as 5 frentes da Aula 2, em profundidade

**Decisão:** cobrir as 5 frentes de estatística descritiva ensinadas na Aula
2 sobre os atributos quantitativos (após excluir Protocol/Dst Port, item 3),
com o máximo de detalhe viável - a EDA é a etapa que mais alimenta o
entendimento do problema, então vale aprofundar mesmo que o notebook fique
grande (o objetivo é visualizar ao máximo o dataset):

- **Frequência:** contagem de `Label` (distribuição de classes), tabela de
  frequência de `Protocol`/`Dst Port` (qualitativos), e cruzamento de
  `Protocol`/`Dst Port` com o alvo (top 10 portas por classe).
- **Localidade:** média, mediana e quartis via `describe()`, mais média,
  mediana e desvio padrão calculados separadamente por classe (BENIGN vs.
  ataque) para os 12 atributos-chave - a comparação que importa de verdade
  para um projeto de classificação.
- **Dispersão:** desvio padrão, intervalo (max-min), IQR (Q3-Q1) e skewness
  (assimetria), via `describe()` estendido; ranking dos 15 atributos mais
  assimétricos entre os 76.
- **Distribuição (forma):** boxplot e histograma de 12 atributos-chave (`Flow
  Duration`, `Total Fwd Packet`, `Flow Bytes/s`, `Flow Packets/s`, `Packet
  Length Mean`, `SYN Flag Count`, `Fwd/Bwd Packet Length Mean`, `Flow IAT
  Mean`, `FWD Init Win Bytes`, `Down/Up Ratio`, `ACK Flag Count`) - cobrindo
  famílias diferentes de atributo -, repetido também agrupado por classe
  (BENIGN vs. ataque lado a lado em cada painel).
- **Correlação:** matriz de correlação de Pearson (amostra de 200k fluxos);
  histograma da distribuição dos 2.850 coeficientes par-a-par (76 escolhe 2),
  justificando o limiar de 0,95 usado na seleção do item 7 (61 pares, 2,14%,
  já ultrapassam esse limiar); correlação de Pearson de cada atributo com o
  próprio alvo (`Label_binario`); pair plot de 6 atributos-chave colorido por
  classe (amostra de 15k fluxos, técnica de visualização multivariada citada
  na Aula 2 como alternativa ao heatmap quando a forma da relação entre pares
  importa).

**Outlier via IQR - testado, não especulado:** contagem de valores fora de
`[Q1 − 1,5×IQR, Q3 + 1,5×IQR]` por atributo (regra do boxplot modificado,
Aula 2) varia de 0% (`FWD Init Win Bytes`, cujo valor máximo possível - 65535,
limite de um campo de 16 bits - já fica abaixo do limite superior calculado)
a ~24% (`Idle Mean/Max/Min`, `Active Mean/Max/Min`, `Flow/Fwd/Bwd
Packets/s`). A primeira versão deste documento especulava que esse outlier
"provavelmente" era o próprio sinal de ataque que o projeto quer detectar -
**a verificação feita depois mostrou que isso só é verdade para alguns
atributos, não em geral.** Calculamos, para cada um dos
12 atributos-chave, a proporção de ataque *entre* os outliers vs. a proporção
de ataque na base inteira (19,1%) - resultado em
`resultados/metricas/outliers_enriquecimento_ataque.csv`:

- `Packet Length Mean` e `Bwd Packet Length Mean` têm enriquecimento de
  4,14× e 4,20× (78,9% e 80,1% dos outliers desses atributos são ataque) -
  aqui sim, outlier é majoritariamente sinal de ataque.
- `Flow Duration`, `Total Fwd Packet`, `SYN/ACK Flag Count`, `Fwd Packet
  Length Mean` e `Flow IAT Mean` têm enriquecimento **abaixo de 1,0** (0,10×
  a 0,53×) - outlier nesses atributos é *menos* provável de ser ataque que a
  média da base, é majoritariamente cauda longa de tráfego benigno normal
  (conexão persistente, rajada legítima).

Não foram removidos como erro: a Aula 2 é explícita que outlier "nem sempre é
erro, pode ser um caso legítimo, só raro", e a informação de separação entre
classes está concentrada de forma desigual entre os atributos, não distribuída
uniformemente pelo "outlier" em geral.

**Por que visualizar além do `describe()`:** a Aula 2 fecha a seção de dados
multivariados com o aviso do Quarteto de Anscombe e do Datasaurus Dozen -
estatísticas resumidas (média, variância, correlação) podem ser idênticas
para dados com formas completamente diferentes; "nunca confiar só nos
números, sempre visualizar". Por isso o pipeline gera boxplot, histograma e
pair plot além da tabela numérica, em vez de reportar só `describe()` e a
matriz de correlação - e, sempre que possível, quebrado por classe, não só
agregado.

**Referência:** Aula 2 (arquivo citado acima), seções "Estatística
descritiva", "Visualização univariada" e "Dados multivariados".

## 7. Seleção de atributos por correlação

**Decisão:** entre os atributos quantitativos (Protocol/Dst Port de fora,
item 3), descartar os com correlação de Pearson > 0,95 (em módulo) com outro
atributo já mantido - 27 de 76 candidatos removidos (lista completa em
`resultados/metricas/matriz_correlacao.csv` e no log de
`04_selecao_e_split.py`), restando 49 quantitativos + 2 qualitativos
(Protocol, Dst Port) = **51 atributos finais**.

**Por quê:** é a aplicação direta de **redundância de atributo**, ensinada na
Aula 4 com o mesmo raciocínio usado aqui: "um atributo cujo valor dá pra
deduzir a partir de outro, porque os dois estão fortemente correlacionados"
(o slide usa número de internações vs. número de visitas). A Aula 4 também
apresenta as duas abordagens de redução de dimensionalidade - **agregação**
(ex. PCA, que combina atributos em componentes sem significado físico direto)
e **seleção de atributos** (mantém os atributos originais, só descarta
redundantes). Optou-se pela segunda: no mesmo espírito da etapa de "feature
extraction" que Talukder et al. (2024) usam via PCA para lidar com a alta
dimensionalidade do CICIDS2017, mas com filtro de correlação em vez de PCA,
porque preserva os atributos originais e é mais fácil de justificar no
relatório da disciplina (que pede para descrever a seleção de atributos, não
componentes abstratos).

**Referência:** Aula 4, seções "Dados inconsistentes e redundantes" e
"Redução de dimensionalidade"; TALUKDER, Md. Alamin et al. *Machine
learning-based network intrusion detection for big and imbalanced data using
oversampling, stacking feature embedding and feature extraction*. Journal of
Big Data, 2024.

**Addendum 23/09/2026 - comparação quantitativa com PCA:** a pedido do
professor em aula, o próprio `04_selecao_e_split.py` (função
`comparar_com_pca`, passo 5 do docstring) roda PCA sobre os 49 atributos
quantitativos já padronizados (amostra de 300k, mesmo critério de custo do
filtro de correlação) só para checar com número real se a escolha acima
(correlação em vez de PCA) se sustenta. Resultado: 22 componentes explicam
90% da variância e 27 explicam 95% (de 49 atributos originais) -
`resultados/metricas/pca_variancia_explicada.csv`,
`resultados/figuras/pca_variancia_explicada.png`. Como o ganho de PCA sobre o
filtro de correlação é pequeno (22-27 componentes abstratos vs. 49 atributos
já reduzidos e interpretáveis), a decisão original não muda: PCA não compensa
nesta etapa, mas fica registrado como alternativa a reconsiderar na etapa de
Análise se o custo computacional do treino exigir mais redução. É um
relatório a mais gerado no final do mesmo script - não altera X_train/
X_test/scaler nem o conjunto final de 51 atributos usado no resto do
pipeline.

**Addendum 15/11/2026 - desempenho com e sem PCA:** `13_comparacao_pca.py`
treina o modelo escolhido para o estágio 1 (Random Forest com undersampling)
trocando os 49 quantitativos por 27 ou 22 componentes do PCA (Protocol e Dst
Port passam sem mudança; o PCA é ajustado dentro de cada fold), no mesmo
protocolo da seção 10. Resultados em `resultados/metricas/estagio1_pca/`.

| Média dos 5 folds | 51 atributos | PCA 27 | PCA 22 |
|---|---|---|---|
| F1 do ataque | 0,9875 | 0,9860 | 0,9859 |
| Ataques perdidos por fold | 26 | 321 | 328 |
| Falsos alarmes por fold | 1.448 | 1.324 | 1.332 |
| Treino por fold | 28 s | 70 s | 57 s |
| Modelo | 19 MB | 37 MB | 38 MB |

O PCA perde: deixa passar 12 vezes mais ataques, treina 2 a 2,5 vezes mais
devagar e gera um modelo com o dobro do tamanho. A diferença é significativa
(McNemar p < 10⁻⁷⁰). A redução de falsos alarmes (cerca de 9%) não compensa.
As perdas se concentram em PortScan (1.523 fluxos perdidos contra 95, somando
os folds) e nas classes raras: nenhum dos 10 SQL Injection do treino é
detectado com PCA (7 sem PCA), e só 11 dos 26 Infiltration (21 sem PCA).
Explicação provável (não testada à parte): o PCA guarda as direções de maior variância, que são as do
tráfego benigno (80% dos fluxos); o que distingue um ataque raro fica em
direções de pouca variância, descartadas junto com os últimos componentes.
Além disso, o SHAP sobre componentes não diria quais atributos pesaram. A
seleção por correlação fica confirmada também pelo desempenho.

## 8. Divisão treino/teste estratificada + padronização sem vazamento

**Decisão:** split 80/20 estratificado pelo rótulo multiclasse (garante que
classes raríssimas, ex. Heartbleed com 11 fluxos, apareçam nos dois
conjuntos); `StandardScaler` ajustado só no treino e aplicado ao teste.

**Por quê:** a padronização - $(v-\text{média})/\text{desvio padrão}$ - é
exatamente a técnica de **normalização por padronização** da Aula 4,
escolhida em vez da normalização por reescala (`min-max`) porque "lida melhor
com outlier que a reescala" (relevante aqui, dado o volume de outlier
identificado no item 6). O ajuste do scaler somente no treino evita
vazamento de dados (data leakage) do conjunto de teste.

**Referência:** Aula 4, seção "Transformação de dados"; KELLEHER, John D.;
TIERNEY, Brendan. *Data Science*. MIT Press, 2018 - capítulo sobre avaliação
de modelos e vazamento de dados.

## 9. Desenho da classificação: hierárquico em dois estágios, com multiclasse direto de referência

**Decisão:** o pipeline principal tem dois estágios. O primeiro é um
classificador binário (BENIGN × ataque) treinado em todo o conjunto de
treino. O segundo classifica o tipo de ataque (14 classes), é treinado só nos
fluxos de ataque do treino e, na avaliação, recebe só os fluxos que o
primeiro estágio marcou como ataque. A previsão final, em 15 classes, é
BENIGN quando o primeiro estágio diz BENIGN e a classe do segundo estágio nos
demais casos. Além disso, um classificador multiclasse direto (15 classes,
estágio único) é treinado com o mesmo algoritmo escolhido para o pipeline
principal, só como referência de comparação.

**Por quê:** reproduz a triagem de um centro de operações de segurança
(detectar primeiro, identificar o tipo depois), desenho já usado em trabalhos
de NIDS em dois estágios. O primeiro estágio é o próprio modelo binário, então
a H1 da proposta é testada nele, e a H2 (multiclasse mais difícil que
binário) compara o primeiro estágio com a previsão final em 15 classes. O
segundo estágio treina em cerca de 234 mil fluxos por fold em vez de 1,2
milhão (Random Forest: 8 s, contra 105 s do multiclasse direto, medido em
14/10/2026), o que viabiliza algoritmos que não escalam para o treino
inteiro, como o kNN cogitado pelo grupo; é o balanço entre custo
computacional e acurácia da Aula 4 ("Amostragem de dados"). Também encaixa
na extensão com RAG proposta como trabalho futuro
(`PROPOSTA-RAG-EXPLICACAO-ALERTAS.md`): só geraria alerta o fluxo que o
primeiro estágio marcou como ataque, e a classe que alimenta o retriever viria
do segundo estágio. O multiclasse direto
existe para que o ganho do hierárquico seja medido, e não presumido.

Duas limitações a discutir no relatório: tirar o BENIGN do segundo estágio
não resolve o desbalanceamento entre os próprios ataques (Heartbleed, 9
fluxos de treino, contra 126.775 de DoS Hulk), que continua a cargo da
decisão de balanceamento; e um fluxo benigno marcado como ataque pelo
primeiro estágio recebe obrigatoriamente uma classe de ataque no segundo.

**Referência:** *Host-Based Network Intrusion Detection via Feature
Flattening and Two-stage Collaborative Classifier*, arXiv:2306.09451, 2023
(classificador binário seguido de multiclasse só nos ataques detectados);
hipóteses H1 e H2 (início deste documento); Aula 4 -
Seleção de Dados, seção "Amostragem de dados".

## 10. Protocolo de avaliação: K-fold estratificado (K=5), F1 como métrica principal, McNemar

**Decisão:**
- **Validação:** K-fold estratificado com K=5, só no conjunto de treino,
  para comparar algoritmos, estratégias de balanceamento e hiperparâmetros.
  Tudo o que é ajustado aos dados (balanceamento e os dois estágios do
  pipeline) é ajustado só na parte de treino de cada fold. O conjunto de
  teste da Seção 8 é usado uma única vez, no modelo final reajustado com o
  treino inteiro.
- **Métrica principal de seleção:** F1 da classe ataque no primeiro estágio;
  F1 macro (média simples do F1 das 15 classes) na previsão final.
- **Métricas reportadas além da principal:** todas as mostradas em aula -
  matriz de confusão, acurácia (sempre ao lado da baseline da classe
  majoritária, 80,9% no binário), precisão, revocação, especificidade,
  ROC/AUC no binário, e FNR e TSS (revocação + especificidade − 1) da
  avaliação multi-métrica da Aula 1. Métricas globais como média e desvio
  padrão dos 5 folds; métricas por classe calculadas sobre as previsões fora
  do fold reunidas (cada fluxo do treino recebe exatamente uma previsão),
  porque cada fold de validação tem só 1 ou 2 fluxos das classes mais raras.
- **Teste estatístico:** McNemar entre pares de algoritmos, sobre o acerto ou
  erro de cada fluxo nas previsões fora do fold, com o tamanho da diferença
  reportado junto do p-valor.

**Por quê:** o slide "Resumo" de Análise de Desempenho de Modelos pede três
coisas que um holdout único não entrega: reportar média e desvio padrão do
desempenho, comparar algoritmos com teste estatístico e reservar o conjunto
de teste só para teste. O K-fold estratificado atende às duas primeiras
mantendo a proporção das classes em cada fold (Kohavi, 1995). K=5 é um dos
dois valores usuais no compromisso entre viés e variância da estimativa
(James et al., 2021) e é o maior K que ainda estratifica Heartbleed, que tem
9 fluxos no treino; também deixa 7 deles no treino de cada fold, acima dos 6
que o SMOTE exige com o padrão de 5 vizinhos. O custo é de cerca de 7 minutos
por configuração no primeiro estágio.

A métrica principal é o F1 porque, com BENIGN dominando a base, acurácia e
F1 ponderado ficam perto de 1 para quase qualquer modelo e escondem o
desempenho nas classes raras; a média macro dá o mesmo peso a cada classe
(Sokolova & Lapalme, 2009), de modo que Heartbleed pesa tanto quanto DoS
Hulk. É também a métrica que os trabalhos recentes sobre o CIC-IDS2017
adotam pelo mesmo motivo, o que permite comparar resultados, e é a média
harmônica da precisão e da revocação do slide. A baseline da classe
majoritária vem da Aula 4 ("a acurácia deve ser maior que a obtida atribuindo
um novo objeto à classe majoritária"); FNR e TSS vêm da Aula 1, que mostra o
modelo de maior ROC AUC sendo o menos confiável, por isso nenhuma métrica é
usada sozinha.

O McNemar compara dois classificadores pelos fluxos em que um acerta e o
outro erra, e é o teste recomendado por Dietterich (1998) para bases grandes
em que repetir o treino muitas vezes é caro; aqui ele usa as previsões que o
K-fold já produz, sem treino adicional. Com 1,2 milhão de fluxos, diferenças
mínimas já dão significativas, por isso o tamanho da diferença acompanha o
p-valor.

**Referência:** slides de Análise de Desempenho de Modelos (seções
"Amostragem", "Métricas para classificação", "Análise ROC" e "Resumo");
Aula 4, seção "Dados desbalanceados"; MIYAJI, VALENCIA DE ALMEIDA & CORRÊA
(2026), estudo de caso da Aula 1; KOHAVI (1995); JAMES et al. (2021), seção
5.1; SOKOLOVA & LAPALME (2009); DIETTERICH (1998); *Do Transformers Actually
Help Intrusion Detection? A Temporal Sequence Evaluation on CIC-IDS2017*,
arXiv:2606.11098, 2026; *When Individually Calibrated Models Become
Collectively Miscalibrated*, arXiv:2605.18858, 2026.

## 11. Classificadores: Random Forest primeiro, kNN e demais se houver tempo

**Decisão (11/11/2026):** o algoritmo do pipeline é o Random Forest (100
árvores), nos dois estágios e no multiclasse direto de referência. O kNN
entra como segundo candidato se o tempo permitir, com subamostra do treino
(o kNN não escala para 1,2 milhão de fluxos); Extra Trees e XGBoost vêm
depois, nessa ordem de prioridade.

**Por quê:** o Random Forest tem respaldo de aula (estudo de caso da Aula 1),
treina o conjunto inteiro em tempo viável (84 s por fold no binário, medido
em 14/10/2026), é multiclasse por construção, aceita `class_weight` e tem
explicador exato e rápido no SHAP (seção 14). Isso o torna o ponto de partida
mais barato para fechar o pipeline completo antes de comparar alternativas.

**Referência:** BREIMAN, Leo. *Random Forests*. Machine Learning, v. 45,
2001; MIYAJI, VALENCIA DE ALMEIDA & CORRÊA (2026), estudo de caso da Aula 1;
Aula 4, seção "Amostragem de dados" (kNN com subamostra).

## 12. Balanceamento: estratégias comparadas por estágio

**Decisão (11/11/2026):** as estratégias são comparadas por estágio, todas
aplicadas só na parte de treino de cada fold (etapa do Pipeline do
imbalanced-learn, ver `modelagem_comum.py`).
- **Estágio 1** (80,9% benigno, 19,1% ataque): sem tratamento, `class_weight`
  balanceado e undersampling aleatório da classe benigna.
- **Estágio 2** (Heartbleed com 9 fluxos de treino contra 126.775 de DoS
  Hulk): sem tratamento, `class_weight` balanceado e SMOTE nas classes raras.
- Se o tempo apertar, fica só o SMOTE no estágio 2.

**Por quê:** o desbalanceamento do estágio 1 é moderado, então reamostrar
1,2 milhão de fluxos com SMOTE custaria caro para pouco ganho. No estágio 2 a
diferença entre classes passa de quatro ordens de grandeza e é onde a
sugestão do professor (aula de 30/09/2026) se aplica. O SMOTE exige mais
vizinhos do que a classe tem de amostras (padrão de 5 vizinhos, mínimo de 6
fluxos), e é por isso que o K=5 da seção 10 deixa 7 fluxos de Heartbleed no
treino de cada fold.

**Referência:** CHAWLA et al. *SMOTE: Synthetic Minority Over-sampling
Technique*. JAIR, v. 16, 2002; Aula 4, seção "Dados desbalanceados";
TALUKDER et al. (2024), já citado na proposta.

## 13. Spark: a H4 é respondida com custo medido em máquina única

**Decisão (11/11/2026):** não há script de Spark. A H4 (o volume impõe
restrições que justifiquem Big Data) é avaliada com o tempo de treino, o
tempo de previsão e o tamanho do modelo que `modelagem_comum.py` já mede em
cada fold, numa máquina de 8 núcleos e 15 GB de RAM. A conclusão e a limitação
(um Spark local não é um cluster) entram no relatório, citando Faker & Dogdu
(2019). Se sobrar tempo ao final, um único Random Forest do Spark MLlib
serve de ponto de comparação de tempo.

**Por quê:** o treino (1,2 milhão de fluxos, float32) cabe na memória e o
Random Forest binário leva 84 s por fold, então o volume não exigiu
processamento distribuído. Spark em modo local, na mesma máquina, não
demonstraria o ganho de um cluster.

**Referência:** FAKER, Ozlem; DOGDU, Erdogan. *Intrusion Detection Using Big
Data and Deep Learning Techniques*. ACM Southeast Conference, 2019; hipótese
H4 (início deste documento); Aula 1 (o "V" de Volume).

## 14. Validação com XAI: SHAP global e local

**Decisão (11/11/2026):** a Fase 3 usa SHAP (TreeSHAP) sobre o modelo final,
em uma amostra estratificada do conjunto de teste que inclui as classes
raras, em vez do teste inteiro. A análise global compara os atributos mais
importantes com a EDA (decisão 6: correlação com o alvo e enriquecimento de
outliers) para responder se o modelo aprendeu padrão de ataque ou artefato do
laboratório. A explicação local por fluxo fica salva e seria a entrada da
query na extensão com RAG (trabalho futuro). Os gráficos de dependência
parcial (PDP) não foram feitos.

**Por quê:** o professor sugeriu o SHAP na aula de 11/11/2026. O TreeSHAP é
exato e rápido para Random Forest, e o estudo de caso da Aula 1 usa o mesmo
tipo de validação.

**Referência:** LUNDBERG & LEE. *A Unified Approach to Interpreting Model
Predictions*. NeurIPS, 2017; LUNDBERG et al. *From local explanations to
global understanding with explainable AI for trees*. Nature Machine
Intelligence, v. 2, 2020; MIYAJI, VALENCIA DE ALMEIDA & CORRÊA (2026).

## 15. Modelos escolhidos e resultados (11/11/2026)

Escolhas feitas pelo K-fold (K=5, só no treino), sem olhar o teste. Scripts 05
a 09; números em `resultados/metricas/`.

| Peça | Escolhido | Critério | Validação cruzada |
|---|---|---|---|
| Estágio 1 | Random Forest com undersampling do BENIGN | F1 do ataque empata com o `class_weight` na quarta casa (0,9875); maior revocação (0,9996) e treino em 28 s contra 95 s | F1 0,9875, AUC 0,9998 |
| Estágio 2 | Random Forest com SMOTE nas classes raras | maior F1 macro da previsão final (0,9307), diferença para as outras duas dentro do ruído (McNemar p = 1,0) | F1 macro 0,931 (final) |
| Multiclasse direto | Random Forest com `class_weight` | maior F1 macro (0,9419 contra 0,9159) | F1 macro 0,942 |

**Conjunto de teste (uma única avaliação):**

| | Resultado |
|---|---|
| Estágio 1 (H1) | F1 0,9867; revocação 0,9995; especificidade 0,9938; AUC 0,9998; 34 ataques perdidos em 72.999 e 1.927 falsos alarmes em 309.607 benignos |
| Hierárquico, 15 classes | acurácia 0,9949; F1 macro 0,9209; precisão macro 0,9267; revocação macro 0,9169 |
| Multiclasse direto, 15 classes | acurácia 0,9949; F1 macro 0,9421; precisão macro 0,9866; revocação macro 0,9171 |
| McNemar hierárquico × direto | 15 fluxos só o hierárquico acerta, 27 só o direto; p = 0,090 |

**Candidatos extras do estágio 1 (script 11, validação cruzada K=5 no treino, `resultados/metricas/estagio1_extras/`):**

| Candidato | F1 | Revocação | Treino por fold | Previsão por fold | McNemar contra o RF |
|---|---|---|---|---|---|
| Random Forest + undersampling (escolhido) | 0,9875 | 0,9996 | 28 s | 0,9 s | - |
| XGBoost + undersampling | 0,9875 | 0,9997 | 5,5 s | 0,5 s | p = 0,71 (empate) |
| Extra Trees + undersampling | 0,9862 | 0,9946 | 36 s | 1,1 s | p = 2e-58 (pior) |
| kNN com subamostra | 0,9832 | 0,9981 | 0,3 s | 112 s | p ≈ 0 (pior) |

**Decisão (14/11/2026): o Random Forest continua no estágio 1.** O XGBoost empata em desempenho e treina mais rápido, mas trocar obrigaria a refazer os scripts 06, 08 e 09 (estágio 2, avaliação final e SHAP) sem ganho de F1. O Extra Trees e o kNN ficam atrás de forma estatisticamente significativa; o kNN ainda custa 112 s de previsão por fold. Os outros três algoritmos também foram testados nos estágios 2 e multiclasse direto (tabela abaixo).

**Candidatos extras do estágio 2 e do multiclasse direto (script 12, validação cruzada K=5 no treino, `resultados/metricas/estagio2_extras/` e `multiclasse_direto_extras/`):**

| Peça | Candidato | F1 macro | Treino por fold | Previsão por fold | McNemar contra o RF |
|---|---|---|---|---|---|
| Estágio 2 (previsão final) | RF + SMOTE (escolhido) | 0,9307 | 10 s | 1,1 s | - |
| | Extra Trees + SMOTE | 0,9383 | 14 s | 1,3 s | p = 0,039 (1 contra 8 fluxos) |
| | XGBoost + SMOTE | 0,9302 | 32 s | 1,2 s | p = 1,0 (7 contra 7) |
| | kNN com subamostra | 0,9181 | 0,1 s | 42 s | p ≈ 0 (pior) |
| Multiclasse direto | RF + class_weight (escolhido) | 0,9419 | 43 s | 2,4 s | - |
| | XGBoost + pesos balanceados | 0,9458 | 151 s | 3,4 s | p = 2e-4 (25 contra 60 fluxos) |
| | Extra Trees + class_weight | 0,8972 | 104 s | 2,5 s | p ≈ 0 (pior) |
| | kNN com subamostra | 0,7260 | 0,3 s | 104 s | p ≈ 0 (pior) |

**Decisão (14/11/2026): o Random Forest continua nos três pontos.** No estágio 2, o Extra Trees tem F1 macro 0,008 acima, mas a diferença é de 7 fluxos em 1,2 milhão e o desvio entre folds é de 0,06, então está dentro do ruído. No multiclasse direto, o XGBoost fica 0,004 acima em F1 macro (35 fluxos de diferença) e treina 3,5 vezes mais devagar. O teste já foi avaliado uma vez com o RF (seção 15), e trocar de modelo depois de olhar esses números exigiria uma segunda avaliação do teste. O kNN é o pior nos dois e o mais lento na previsão. Conclusão para o relatório: com este dataset e esses atributos, a escolha do algoritmo de árvores importa pouco; o gargalo é a detecção das classes raras no estágio 1.

**Classes raras (XSS, SQL Injection, Infiltration, Web Attack Brute Force):** tratadas como discussão e limitação no relatório, sem nova rodada de modelagem. O F1 baixo vem da detecção do estágio 1 (taxa de detecção de 0,60 no XSS, 0,50 no SQL Injection, 0,83 no Brute Force e na Infiltration) e o teste tem de 2 a 30 fluxos nessas classes, então um fluxo errado muda o F1 em dezenas de pontos. O SMOTE (sugestão do professor de 30/09) já está no estágio 2; o gargalo é o estágio 1, que usa undersampling do BENIGN e não vê essas classes como ataque. Caminhos naturais, não executados: coletar mais exemplos ou usar outra base com mais ataques web.

**Leitura por hipótese:**
- **H1 (detecção binária):** confirmada. F1 de 0,987 e AUC de 0,9998 no teste, iguais aos da validação cruzada.
- **H2 (multiclasse mais difícil que o binário):** confirmada em F1 macro (0,92 a 0,94 contra 0,987). O hierárquico não superou o direto: a acurácia é a mesma e a diferença no McNemar não é significativa. A revocação macro é idêntica (0,917); os 0,02 de diferença no F1 macro são 3 fluxos: o hierárquico dá um falso alarme a mais em SQL Injection (2 fluxos no teste, precisão 0,50 contra 1,00), XSS (5 fluxos, 0,75 contra 1,00) e Infiltration (6 fluxos, 0,83 contra 1,00), e o F1 macro pesa cada classe igual. A precisão da PortScan é a mesma nos dois (0,904): os falsos alarmes do estágio 1 caem quase todos em PortScan (1.923 de 1.927), mas o direto confunde benigno e PortScan na mesma medida, então isso não vem do desenho em dois estágios. O hierárquico treina mais rápido (estágio 2 em 10 s por fold contra 43 s do direto) e entrega a classe que alimentaria o RAG.
- **H3 (classes raras):** o F1 das classes raras é menor e instável. No teste: Heartbleed 1,00 (2 fluxos), Infiltration 0,83 (6), XSS 0,67 (5), SQL Injection 0,50 (2), Web Attack Brute Force 0,88 (30). Com tão poucos fluxos o número é anedótico. O erro nasce na detecção: o estágio 1 deixa passar 17% do Brute Force, 40% do XSS e 50% do SQL Injection. O desvio padrão do F1 macro entre folds (0,06) vem das mesmas classes.
- **H4 (volume e Big Data):** respondida com custo medido em uma máquina (8 núcleos, 15 GB), conforme a seção 13. Por fold, o estágio 1 com undersampling treina em 28 s e prevê em 0,9 s (modelo de 19 MB); o estágio 2 treina em 10 s (5 MB); o multiclasse direto treina em 43 s (36 MB). Nenhum precisou de processamento distribuído.

**SHAP (seção 14):** amostra estratificada de 2.893 fluxos do teste.
- Atributos mais importantes no estágio 1: `Bwd Packet Length Mean` (0,062), `Dst Port` (0,062), `Bwd Packet Length Max`, `RST Flag Count`, `Bwd Init Win Bytes`, `SYN Flag Count`.
- Comparação com a EDA (Spearman): a importância SHAP concorda com a correlação com o alvo (ρ = 0,60, n = 48, p = 6e-6) e não concorda com o enriquecimento de outliers (ρ = −0,13, n = 9, p = 0,73; só 9 atributos entram nessa comparação).
- A `Dst Port` em segundo lugar é discutida como possível shortcut learning: parte legítima (ataques miram portas específicas), parte possível artefato de um laboratório de topologia fixa. Esta análise não separa as duas.
- Erros na amostra: 13 de 2.893, 12 deles ataques classificados como BENIGN (5 Brute Force, 2 XSS, 2 PortScan, e 1 de cada: slowloris, SQL Injection, Infiltration).
- Explicações locais por fluxo (2.581 fluxos marcados como ataque) em `resultados/metricas/xai/explicacoes_locais.csv`.

**Limitações a declarar no relatório:** poucos fluxos nas classes raras no teste
(2 a 6); precisão do estágio 1 de 0,974 (1.927 falsos alarmes no teste);
fluxo benigno marcado como ataque recebe um tipo de ataque; dataset de um único
laboratório (CICIDS2017 corrigido), sem validação em outra base.

## Trabalho futuro

- **Explicação dos alertas com RAG:** proposta detalhada em
  `PROPOSTA-RAG-EXPLICACAO-ALERTAS.md`, não implementada.
- **Spark MLlib:** um Random Forest do Spark como ponto de comparação de tempo
  com o pipeline de máquina única (seção 13).
- **Classes raras:** mais exemplos de ataques web e de Infiltration, ou
  validação em outra base, para que o F1 dessas classes deixe de ser
  anedótico (seção 15).
- **Shortcut learning na `Dst Port`:** separar a parte legítima da parte que
  é artefato do laboratório (seção 15).

## Bibliografia

- BREIMAN, Leo. *Random Forests*. Machine Learning, v. 45, 2001.
- CHAWLA, N. V. et al. *SMOTE: Synthetic Minority Over-sampling Technique*.
  Journal of Artificial Intelligence Research, v. 16, 2002.
- DIETTERICH, Thomas G. *Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms*. Neural Computation, v. 10,
  n. 7, p. 1895-1923, 1998.
- ENGELEN, Gints; RIMMER, Vera; JOOSEN, Wouter. *Troubleshooting an Intrusion
  Detection Dataset: the CICIDS2017 Case Study*. IEEE Security and Privacy
  Workshops (SPW) / WTMC, 2021.
- FAKER, Ozlem; DOGDU, Erdogan. *Intrusion Detection Using Big Data and Deep
  Learning Techniques*. ACM Southeast Conference, 2019.
- JAMES, Gareth; WITTEN, Daniela; HASTIE, Trevor; TIBSHIRANI, Robert. *An
  Introduction to Statistical Learning*. 2. ed. Springer, 2021.
- KELLEHER, John D.; TIERNEY, Brendan. *Data Science*. MIT Press, 2018.
- KOHAVI, Ron. *A Study of Cross-Validation and Bootstrap for Accuracy
  Estimation and Model Selection*. IJCAI, 1995.
- LUNDBERG, Scott M.; LEE, Su-In. *A Unified Approach to Interpreting Model
  Predictions*. NeurIPS, 2017.
- LUNDBERG, Scott M. et al. *From local explanations to global understanding
  with explainable AI for trees*. Nature Machine Intelligence, v. 2, 2020.
- MIYAJI, R. O.; VALENCIA DE ALMEIDA, F.; CORRÊA, P. L. P. *Standardized ML
  and XAI Workflow for Presence-Only Species Distribution Modeling: A Case
  Study in the Amazon*. IEEE Access, 2026. DOI 10.1109/ACCESS.2026.3652949.
- SHARAFALDIN, Iman; LASHKARI, Arash Habibi; GHORBANI, Ali A. *Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization*. ICISSP, 2018.
- SOKOLOVA, Marina; LAPALME, Guy. *A systematic analysis of performance
  measures for classification tasks*. Information Processing & Management,
  v. 45, n. 4, p. 427-437, 2009.
- TALUKDER, Md. Alamin et al. *Machine learning-based network intrusion
  detection for big and imbalanced data using oversampling, stacking feature
  embedding and feature extraction*. Journal of Big Data, 2024.
- *Host-Based Network Intrusion Detection via Feature Flattening and
  Two-stage Collaborative Classifier*. arXiv:2306.09451, 2023.
- *Do Transformers Actually Help Intrusion Detection? A Temporal Sequence
  Evaluation on CIC-IDS2017*. arXiv:2606.11098, 2026.
- *When Individually Calibrated Models Become Collectively Miscalibrated*.
  arXiv:2605.18858, 2026.
- Slides da disciplina PTC3567 - Ciência dos Dados em Automação e Engenharia,
  Escola Politécnica da USP, 2026: Aula 1; Aula 2 - Caracterização e Análise
  Exploratória dos Dados; Aula 4 - Seleção de Dados; Análise de Desempenho de
  Modelos. Material interno da disciplina, não redistribuído aqui.
