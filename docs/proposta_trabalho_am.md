# Proposta de Trabalho — Aprendizado de Máquina (AM)

**Projeto:** QA Gen Portal — Estimativas de Esforço, Risco e Perfil de Requisitos  
**Curso:** Pós-Graduação em Inteligência Artificial (INF/UFG)

---

## 1. Contexto
Este trabalho dá continuidade ao projeto desenvolvido nas disciplinas anteriores (*Engenharia de Prompts* e *Framework de Desenvolvimento Web*). O sistema original recebe um requisito funcional em texto e utiliza uma LLM para gerar cenários BDD, casos de teste manuais, scripts Playwright e casos de borda.

Nesta etapa de **Aprendizado de Máquina**, adicionamos um módulo preditivo que extrai indicadores objetivos do requisito processado e apoia o planejamento de testes.

---

## 2. Atributos (Features) Utilizados
Para manter o uso simples e intuitivo tanto na extração automática quanto nas telas de simulação, utilizamos 4 atributos por requisito:
1. `num_regras_negocio`: quantidade de regras de negócio identificadas.
2. `num_casos_teste`: quantidade de casos de teste listados.
3. `num_integracoes`: quantidade de integrações com APIs ou serviços externos mencionadas.
4. `envolve_seguranca`: indicador binário (`0` ou `1`) informando se o requisito trata de autenticação, permissões ou dados sensíveis.

---

## 3. Tarefas Implementadas

### 3.1 Previsão de Esforço (Regressão)
- **Objetivo:** Estimar o tempo necessário (em horas) para testar e homologar o requisito (`esforco_teste_horas`).
- **Modelo:** `RandomForestRegressor`.
- **Avaliação:** Coeficiente de determinação ($R^2$), Erro Médio Absoluto (MAE) e RMSE, além do gráfico de importância das variáveis.

### 3.2 Risco de Defeito em Produção (Classificação Supervisionada)
- **Objetivo:** Prever a probabilidade de o requisito gerar falhas em produção (`gerou_bug_producao`) e classificá-lo em risco *Baixo*, *Médio* ou *Alto*.
- **Modelo:** `LogisticRegression` com padronização via `StandardScaler`.
- **Avaliação:** Acurácia, Precisão, Recall e F1-Score.

### 3.3 Agrupamento de Perfis (Clusterização Não Supervisionada)
- **Objetivo:** Identificar grupos naturais de requisitos sem usar rótulos no treinamento.
- **Modelo:** `KMeans` ($k=4$) sobre dados padronizados (`StandardScaler`).
- **Avaliação:** Inércia e Coeficiente de Silhueta, agrupando os requisitos em 4 perfis: *CRUD Simples*, *Fluxo de Negócio*, *Integração Externa* e *Crítico / Segurança*.

---

## 4. Entregáveis no Repositório
- **Aplicação Web (`portal/`):** Aba "Estimativas (ML)" na visualização de cada requisito e páginas dedicadas em `/ml/regressao/`, `/ml/classificacao/` e `/ml/clusterizacao/`.
- **Base de Dados (`data/historico_requisitos.csv`):** 300 registros históricos simulados para treinamento e validação.
- **Notebook (`notebooks/AM_QA_Gen_Insights.ipynb`):** Experimento prático demonstrando a leitura do CSV, treinamento e avaliação das três tarefas.
