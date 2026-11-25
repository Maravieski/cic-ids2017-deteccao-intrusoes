# Proposta: RAG para explicar os alertas do classificador

> **Trabalho futuro, não implementado.** Esta proposta foi desenhada durante o
> projeto e ficou fora do escopo final. Fica registrada como ponto de partida
> para quem quiser estender o pipeline.

Origem: sugestão do professor da disciplina de integrar RAG ao projeto e
documentar cada escolha. Foi a ideia escolhida pelo grupo entre as levantadas.
Ao ser implementada, cada escolha abaixo viraria uma seção numerada de
`DECISOES-E-BIBLIOGRAFIA.md`, no formato Decisão / Por quê / Referência.

**Pontos a rever antes de implementar**, a partir de um exercício anterior de
RAG feito na disciplina, sobre outra base de documentos:
- Itens 3 e 7 (modelos locais via Ollama): no exercício, o `nomic-embed-text`
  recuperou mal e um LLM local de 3B em CPU se mostrou lento e fraco. A
  alternativa que funcionou foi o `BAAI/bge-m3` para embeddings, com um LLM
  hospedado. Reavaliar o argumento de privacidade (dados locais) diante das
  opções que sobram.
- Item 5 (retriever): o exercício deu evidência concreta. Com k=3, o MMR
  acertou as 5 perguntas de teste e a similaridade pura não; a similaridade
  chegou a recuperar o termo certo no período errado, o que reforça o filtro
  por metadado.
- Item 6 (query): o pipeline já gera explicação local por fluxo com SHAP
  (seção 14 de `DECISOES-E-BIBLIOGRAFIA.md`, arquivo
  `resultados/metricas/xai/explicacoes_locais.csv`). A query pode usar os
  atributos que mais pesaram naquele fluxo específico, em vez da importância
  global do Random Forest, o que resolve o risco "explicação por importância
  global".

## A ideia em uma frase

O classificador continua sendo quem detecta. O RAG entra depois dele e
transforma a saída ("fluxo 18.342: DoS slowloris, 91%") num alerta que um
analista consegue ler: o que é esse ataque, por que esse fluxo se parece com
ele e o que fazer.

## Por que faz sentido neste projeto

- **Resolve uma limitação real de NIDS com ML:** o modelo devolve um rótulo e
  uma probabilidade, sem contexto. Num SOC, o analista ainda precisa ir atrás
  do que aquele rótulo significa. É a parte "acionável" que falta no pipeline.
- **Não compete com o classificador:** as métricas de detecção (F1, recall por
  classe) não mudam. O RAG é uma camada a mais, então não arrisca o resultado
  principal do relatório.
- **Funciona bem justamente nas classes raras:** Heartbleed (11 fluxos), SQL
  Injection (12) e Infiltration (32) quase não têm amostras para o modelo
  aprender, mas têm tanta documentação quanto qualquer outro ataque. O
  conhecimento sobre o ataque não depende de quantos fluxos existem dele.
- **Pipeline padrão:** loader, splitter, embeddings, Chroma, retriever e
  chain, como num RAG comum; só mudam a base de documentos e o formato da
  pergunta.

## Casos reais na indústria (levantados em outubro de 2026)

- **Microsoft, Guided Response no Defender XDR / Security Copilot:** para
  cada incidente, recomenda ações em quatro categorias (triagem, contenção,
  investigação, remediação), cada uma com a explicação do porquê. A
  organização pode subir as próprias diretrizes para que as recomendações as
  sigam, que é a ideia do playbook interno. Tem artigo descrevendo o sistema,
  em produção em milhares de clientes: arXiv 2407.09017. Doc:
  <https://learn.microsoft.com/defender-xdr/security-copilot-m365d-guided-response>.
  É o caso mais próximo da proposta e o mais citável.
- **Google Security Operations (Gemini):** resumo contextualizado do caso,
  ações de resposta recomendadas e geração de playbook SOAR; o veredito do
  agente de triagem pode disparar um playbook de remediação, com aprovação
  manual para ações sensíveis. O modelo usa playbooks SOAR entre as fontes.
  <https://docs.cloud.google.com/chronicle/docs/secops/gemini-chronicle>,
  <https://docs.cloud.google.com/chronicle/docs/secops/respond-cases>.
- **Elastic Security, AI Assistant:** RAG explícito. Uma Knowledge Base com
  documentos da própria organização e os alertas recentes; o prompt de resumo
  de alerta explica por que o alerta disparou e recomenda passos de triagem e
  remediação. <https://www.elastic.co/docs/solutions/security/ai/ai-assistant-knowledge-base>.

Na academia, o mais próximo é Cadet et al., *Retrieval-Augmented LLMs for
Security Incident Analysis* (arXiv 2603.18196, 2026), que usa RAG guiado por
técnicas do ATT&CK sobre logs; sem RAG, o LLM deixa passar a infraestrutura do
ataque.

## Arquitetura

```
fluxo de teste ──► classificador ──► classe prevista + probabilidade
                                            │
                     atributos do fluxo ────┤
                                            ▼
                               montagem automática da query
                                            │
                                            ▼
              Retriever (filtro por classe + MMR) ◄── Vector DB (Chroma)
                                            │               ▲
                                            ▼               │
                      Prompt (contexto + dados do fluxo)    │
                                            │       documentos sobre ataques
                                            ▼       (chunks + embeddings)
                                   LLM ──► alerta explicado
```

Diferença importante em relação ao diagrama da aula: lá a query vem de um
usuário digitando uma pergunta. Aqui ela é **montada automaticamente** a
partir da saída do classificador, então o sistema roda sem ninguém perguntar
nada, um alerta por fluxo classificado como ataque.

## Escolhas a fazer e justificativa proposta

### Pré-produção

**1. Documentos da base de conhecimento**

| Fonte | Formato | O que contribui |
|---|---|---|
| Sharafaldin et al. (2018), artigo do CIC-IDS2017 | PDF | Como cada ataque foi gerado no laboratório e com qual ferramenta (Hulk, GoldenEye, slowloris, Patator, Ares...) |
| Engelen et al. (2021), versão corrigida que o projeto usa | PDF | Ressalvas de rotulagem por classe (ex. tentativas fundidas em BENIGN, decisão 5) |
| MITRE ATT&CK, técnicas correspondentes às classes | JSON (STIX) | Descrição padronizada, mitigação e detecção recomendadas |
| CVE-2014-0160 (Heartbleed) | texto | Detalhe da vulnerabilidade |
| Perfil estatístico de cada classe, gerado a partir da EDA do próprio projeto | CSV/texto | Valores típicos dos atributos por classe, para o "por que este fluxo se parece com o ataque" |
| Playbooks de resposta a incidente (candidatos a conferir: CISA *Incident and Vulnerability Response Playbooks* (2021), IRM do CERT Société Générale, MITRE D3FEND) | PDF/MD | Passos de resposta por tipo de incidente (conter, erradicar, recuperar), usados na "Ação recomendada" |

Mapeamento inicial classe → ATT&CK (conferir na base oficial antes de usar):
DoS Hulk / GoldenEye / slowloris / Slowhttptest → T1499 (Endpoint Denial of
Service); DDoS → T1498 (Network Denial of Service); PortScan → T1046 (Network
Service Discovery); FTP-Patator, SSH-Patator, Web Attack Brute Force → T1110
(Brute Force); SQL Injection, XSS, Heartbleed → T1190 (Exploit Public-Facing
Application); Bot → T1071 (Application Layer Protocol, canal de C2);
Infiltration → combinação de técnicas, mapear a partir da descrição do
artigo original.

Justificativa: são fontes citáveis (DOI ou URL versionada, o que atende a
gestão de dados do projeto) e cobrem as
quatro perguntas do alerta: o que é (ATT&CK, CVE), como aparece neste dataset
(Sharafaldin, Engelen), como aparece nos atributos (perfil da EDA) e o que
fazer agora (playbook).

**Por que playbook e não só as mitigações do ATT&CK:** as mitigações do ATT&CK são preventivas, de endurecimento do
ambiente ("configure timeouts", "use autenticação multifator"). Quem recebe o
alerta precisa da resposta ao incidente em andamento: o que conter primeiro,
o que coletar, quando escalar. Isso está nos playbooks. É também o caso de uso
mais realista de RAG em segurança: na prática cada organização tem o próprio
playbook interno, e é sobre ele que o RAG buscaria, assim como no exemplo da
aula ele busca a política de reembolso interna da empresa. Os playbooks
públicos fazem esse papel no projeto.

Os playbooks são organizados por tipo de incidente, não por classe do
dataset, então o metadado deles é uma **família** (DoS/DDoS, força bruta,
varredura, exploração de aplicação web, bot/C2), e um mapa fixo classe →
família faz a ponte.

**2. Chunking**

Proposta: tratar cada tipo de documento de um jeito.
- Entradas do ATT&CK e o perfil de cada classe já são unidades naturais
  (uma técnica, uma classe), então cada uma vira **um chunk inteiro**, sem
  corte por tamanho.
- Os artigos passam por `RecursiveCharacterTextSplitter` (~1000 caracteres,
  overlap ~200), que tenta cortar em parágrafo
  antes de cortar no meio da frase.
- Playbooks são cortados **por fase/passo** (contenção, erradicação,
  recuperação), sem quebrar uma lista numerada de procedimento no meio.
- Todo chunk recebe **metadado com a(s) classe(s) a que se refere** (ou a
  família, no caso dos playbooks) e o **tipo de documento** (`conhecimento`
  ou `playbook`).

Justificativa: cortar uma técnica do ATT&CK no meio por tamanho fixo separa a
descrição da mitigação. O metadado de classe é o que permite o filtro do
retriever (item 5), que é a escolha que mais pesa na qualidade.

**3. Modelo de embedding**

Proposta: modelo de embedding rodando local (o `nomic-embed-text` via Ollama
foi a ideia inicial; o `BAAI/bge-m3` recuperou melhor no exercício citado no
topo).

Justificativa: custo zero, roda sem GPU, e os dados de tráfego não saem da
máquina, o que num contexto de segurança é argumento por si só (logs de rede
de uma empresa não deveriam ir para uma API externa). Os documentos são em
inglês, idioma em que o modelo é forte.

**4. Vector database**

Proposta: Chroma, persistido em disco.

Justificativa: a base tem centenas de chunks, não milhões, então não precisa
de nada pensado para escala (FAISS com índices aproximados, pgvector,
serviços gerenciados). Chroma suporta filtro por metadado, que o item 5
precisa.

### Em produção

**5. Retriever**

Proposta: filtro por metadado `classe = classe prevista` e, dentro disso,
busca MMR com k = 3 ou 4.

Justificativa: a similaridade semântica pura confunde ataques parecidos
(os quatro DoS de camada de aplicação têm descrições muito próximas). Como o
classificador já disse qual é a classe, filtrar por ela é mais confiável do
que torcer para o embedding acertar. O MMR evita que os k chunks sejam três
versões do mesmo parágrafo; a comparação similaridade × MMR do exercício
citado no topo serve de evidência para essa escolha.

**Duas buscas separadas** por alerta: uma em `conhecimento` (para "O que é"
e "Por que este fluxo") e outra em `playbook` filtrada pela família (para
"Ação recomendada"). Numa busca única, os chunks de playbook disputariam as
mesmas k vagas com os artigos e poderiam ficar de fora; separando, cada
parte do alerta tem garantido o contexto de que precisa.

Caso de **baixa confiança** (probabilidade abaixo de um limiar, ex. 0,6):
recuperar para as duas classes mais prováveis e o alerta dizer que está em
dúvida entre elas. Isso trata a principal fraqueza da ideia (ver Riscos).

**6. Montagem da query**

A query inclui: classe prevista e probabilidade; porta de destino e
protocolo; os atributos que mais pesaram naquele fluxo segundo o SHAP local,
com o valor do fluxo comparado à mediana da classe e à mediana de BENIGN.

Justificativa: é isso que permite ao LLM explicar "por que este fluxo", e
não só descrever o ataque genericamente. Liga a explicação de volta à EDA
(estatísticas por classe, decisão 6).

**7. LLM do generator**

Proposta: modelo local via Ollama (algo na faixa de 3B a 8B parâmetros, que
caiba nos 15 GB de RAM sem GPU), temperatura 0. Alternativa: Gemini no plano
gratuito, se o local ficar lento demais.

Justificativa: mesma do embedding (custo e privacidade). Temperatura 0
porque um alerta de segurança precisa ser reprodutível, sem variação
criativa entre execuções. Como só os fluxos classificados como ataque geram
alerta, e para o relatório basta uma amostra, a lentidão de rodar em CPU é
aceitável.

**8. Prompt**

Instruções: responder em português; usar só o contexto recuperado; citar a
fonte de cada afirmação; se o contexto não sustentar algo, dizer que não
sabe; saída em formato fixo com quatro partes (Classificação, O que é, Por
que este fluxo, Ação recomendada).

Regra mais rígida para a "Ação recomendada": o LLM só pode listar passos que
estejam no playbook recuperado, citando de qual fase vêm, e pode apenas
preencher os dados do fluxo (porta, protocolo) dentro deles. Nenhum passo
novo. A ação é **recomendada ao analista, nunca executada automaticamente**.

Justificativa: formato fixo facilita a avaliação (item abaixo) e a leitura
pelo analista; restringir ao contexto é o que torna o RAG útil contra
alucinação. Na ação o rigor é maior porque o custo do erro é maior: uma
explicação inventada confunde, mas um passo de resposta inventado (bloquear
uma faixa de IPs, derrubar um serviço) pode parar tráfego legítimo.

## Exemplo de saída (ilustrativo, valores inventados)

> **Classificação:** DoS slowloris (confiança 0,91). TCP, porta 80.
>
> **O que é:** ataque de negação de serviço que abre muitas conexões HTTP e
> as mantém vivas enviando cabeçalhos incompletos aos poucos, até esgotar as
> conexões que o servidor aceita [ATT&CK T1499; Sharafaldin 2018].
>
> **Por que este fluxo:** duração de 98 s com só 7 pacotes e intervalo médio
> entre pacotes muito acima da mediana de BENIGN, que é o padrão de manter
> a conexão aberta com o mínimo de tráfego [perfil da classe].
>
> **Ação recomendada (playbook DoS, fase de contenção):** 1. confirmar com a
> equipe do serviço se há degradação no servidor web da porta 80; 2. limitar
> conexões simultâneas por IP de origem no balanceador/firewall; 3. reduzir
> o timeout de recebimento de cabeçalho; 4. se persistir, escalar para o
> provedor de mitigação de DDoS [playbook DoS, contenção, passos 1 a 4].

## Como avaliar (para a seção de resultados)

1. **Recuperação:** para uma amostra de alertas (ex. 3 por classe de ataque),
   verificar se os chunks recuperados falam da classe certa. Com o filtro por
   metadado isso tende a 100%, então vale medir também **sem** o filtro e
   mostrar a diferença; é uma justificativa quantitativa do item 5.
2. **Geração:** avaliar cada alerta da amostra com uma rubrica simples
   (descrição correta? explicação coerente com os atributos? alguma afirmação
   sem fonte no contexto? todo passo da ação existe no playbook recuperado?).
3. **Com RAG × sem RAG:** gerar as mesmas explicações só com o LLM, sem
   contexto recuperado, e comparar pela mesma rubrica. É o experimento que
   mostra se o RAG agrega algo.

## Riscos e limitações (entram na discussão do relatório)

- **Erro do classificador se propaga:** se o modelo errar a classe, o RAG
  explica com confiança o ataque errado. Mitigação parcial: o tratamento de
  baixa confiança no item 5, e mostrar sempre a probabilidade no alerta.
- **Explicação por importância global:** a feature importance do Random
  Forest é da classe toda, não do fluxo específico. Por isso a query usa o
  SHAP local por fluxo, que o pipeline já calcula.
- **Playbooks públicos são genéricos:** escritos para qualquer organização,
  sem os nomes de sistemas, contatos e ferramentas de um playbook interno
  real. O alerta vai sair genérico nesse ponto, e isso deve ser dito como
  limitação (numa implantação real, a base seria o playbook da própria
  empresa).
- **Avaliação manual e pequena:** a rubrica é qualitativa e feita pelo
  grupo; deixar isso explícito.

## Escopo mínimo

O que basta para o relatório: base com os artigos + ATT&CK, pipeline rodando
sobre uma amostra do conjunto de teste (alguns fluxos por classe), avaliação
dos três itens acima. Consome o modelo final e as explicações locais já
geradas pelos scripts 08 e 09.
