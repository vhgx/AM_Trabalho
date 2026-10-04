# Proposta de Trabalho da Disciplina: Aprendizado de Máquina (AM)

**Título do Trabalho:** QA Gen Insights: Módulo de Machine Learning para Previsão de Esforço, Detecção de Risco e Descoberta de Perfis de Requisitos de Software

**Curso:** Pós-Graduação em Inteligência Artificial &bull; INF / UFG  
**Componentes do Grupo:**
- [Seu Nome Completo]
- [Nome do Componente 2 - se houver]
- [Nome do Componente 3 - se houver]

---

## 1. Contexto e Integração com o Projeto Aplicado (PA)
Este trabalho representa a continuidade e evolução direta das entregas anteriores do curso:
1. **Engenharia de Prompts:** Desenvolveu o núcleo de raciocínio do *QA Test Generator Agent* (tradução de histórias de usuário em BDD, casos de teste manuais e scripts de automação Playwright).
2. **Framework de Desenvolvimento Web:** Estruturou a arquitetura MVC/SaaS com Django (*QA Gen Web Portal*), viabilizando a persistência em banco relacional e a navegação visual dos requisitos.
3. **Aprendizado de Máquina (Entrega Atual):** Acopla uma camada analítica e preditiva (*QA Gen Insights*) capaz de extrair atributos estruturais diretamente dos artefatos produzidos pelos LLMs e responder a 3 desafios do ciclo de vida de QA sem custo computacional adicional de inferência de linguagem natural.

---

## 2. Objetivo e Descrição Funcional
O objetivo da solução é capacitar analistas de qualidade e líderes técnicos a tomarem decisões baseadas em dados históricos a cada novo requisito processado:
- **Previsão Contínua (Regressão):** Estimar antecipadamente o esforço total de homologação (em horas).
- **Classificação Preventiva (Supervisionado Discreto):** Alertar se o requisito possui propensão a introduzir defeitos críticos em produção (*escaped defects*).
- **Agrupamento Autônomo (Não Supervisionado):** Identificar o perfil e arquétipo do requisito (*CRUD Simples*, *Fluxo Complexo*, *Integração Pesada*, *Crítico/Segurança*) para direcionar a estratégia de teste correta.

---

## 3. Escopo de Machine Learning e Modelos Implementados

Conforme os conceitos teóricos e laboratoriais ministrados na disciplina, o sistema adota:

### 3.1 Regressão (Previsão de Esforço em Horas)
- **Modelos:** Comparativo formal entre `LinearRegression` (baseline interpretável) e `RandomForestRegressor` (ensemble não-linear com 100 estimadores).
- **Validação:** Validação Cruzada K-Fold (5 partições aleatórias).
- **Métricas:** Coeficiente de Determinação ($R^2$), Erro Médio Absoluto (MAE) e Raiz do Erro Quadrático Médio (RMSE).
- **Interpretabilidade:** Extração dos coeficientes padronizados ($\beta$) e importância relativa de atributos (*Gini Importance*).

### 3.2 Classificação (Detecção de Risco de Bugs em Produção)
- **Modelos:** Comparativo entre `LogisticRegression` e `DecisionTreeClassifier` (com poda preventiva `max_depth=4`).
- **Validação:** Stratified K-Fold (5 partições preservando a proporção de classes desbalanceadas).
- **Métricas:** Matriz de Confusão, Acurácia, Precisão, Recall (Sensibilidade), F1-Score e Curva ROC com Área sob a Curva (ROC-AUC).
- **Calibração de Limiar (*Threshold Tuning*):** Redução do limiar operacional padrão de $0.50$ para $0.35$, priorizando *Recall* (prevenção contra a perda de defeitos graves em produção).

### 3.3 Clusterização e Redução de Dimensionalidade (Perfis de Requisitos)
- **Modelos:** Algoritmo iterativo `KMeans` sobre atributos padronizados (`StandardScaler`).
- **Determinação de $K$:** Varredura de $K=2$ a $K=8$ auditada pelo *Método do Cotovelo* (Inércia intra-cluster / WCSS) e *Coeficiente de Silhueta* ($-1$ a $+1$), convergindo no $K=4$ ótimo.
- **Redução Dimensional:** `PCA (Análise de Componentes Principais)` retendo ~75% da variância em 2 componentes principais, viabilizando a projeção visual e o mapeamento dos centróides no plano cartesiano.
- **Rotulação Semântica:** Identificação dos perfis *CRUD Simples*, *Fluxo de Negócio Complexo*, *Integração Pesada* e *Crítico / Segurança*.

---

## 4. Engenharia de Atributos (Feature Extraction)
As features de entrada são computadas deterministicamente via expressões regulares a partir dos campos do modelo Django `Requisito`:
1. `num_regras_negocio`: Itens delimitados na interpretação do requisito.
2. `num_cenarios_bdd`: Contagem de keywords `Scenario:` / `Scenario Outline:`.
3. `num_casos_teste`: Linhas úteis da tabela Markdown de casos estruturados.
4. `num_edge_cases`: Contagem de cenários de teste de limites e ruptura.
5. `num_integracoes`: Detecção léxica de APIs, webhooks, gateways, mensageria (Kafka/RabbitMQ) e serviços terceiros.
6. `envolve_seguranca`: Indicador binário (0 ou 1) para login, tokens JWT, criptografia, MFA e LGPD.
7. `tamanho_descricao`: Extensão total da descrição textual em palavras.

---

## 5. Entregas e Artefatos da Disciplina
1. **Web App Operacional (Django 4.2+):**
   - Nova aba *"🤖 Insights de ML"* na tela de resultados de cada requisito (`resultado.html`).
   - Três laboratórios interativos dedicados: `/ml/regressao/`, `/ml/classificacao/` e `/ml/clusterizacao/`.
   - Comando de gestão: `python manage.py treinar_modelos`.
2. **Jupyter Notebook Acadêmico (`notebooks/AM_QA_Gen_Insights.ipynb`):**
   - Condução científica completa, EDA, comparativo de modelos, gráficos de métricas e conclusões.
3. **Base Histórica Replicável (`data/historico_requisitos.csv`):**
   - 400 registros sintéticos calibrados com semente fixa (`seed=42`).
