# 🧪 QA Gen Web Portal & Insights

> **Plataforma Web para Geração Automatizada de Testes e Inteligência Analítica de Qualidade via Aprendizado de Máquina**  
> *Projeto Aplicado &bull; Pós-Graduação em Inteligência Artificial &bull; INF / UFG*

---

## 🎯 Visão Geral

O **QA Gen Insights** é a evolução analítica do portal de engenharia de qualidade. Além de gerar cenários BDD, casos de teste manuais e scripts Playwright via LLM, o sistema agora integra uma camada de **Aprendizado de Máquina Clássico (scikit-learn)** que responde a 3 perguntas essenciais para a tomada de decisão do time:

1. **📈 Regressão:** *Quantas horas de esforço de teste este requisito exigirá?*  
   Modelos: `RandomForestRegressor` ($R^2 \approx 0.93$) e `LinearRegression` ($R^2 \approx 0.96$). Avaliado via K-Fold CV.
2. **🛡️ Classificação:** *Qual o risco deste requisito gerar defeitos em produção?*  
   Modelos: `LogisticRegression` ($\text{AUC} \approx 0.77$) e `DecisionTreeClassifier`. Calibração de limiar ($\tau = 0.35$) focada em maximizar o **Recall**.
3. **🔬 Clusterização & PCA:** *Qual o perfil operacional deste requisito no histórico do time?*  
   Modelos: `KMeans` ($K=4$, Silhueta $\approx 0.48$) com redução de dimensionalidade via `PCA 2D` ($\approx 75\%$ de variância retida). Identifica arquétipos como *CRUD Simples*, *Fluxo Complexo*, *Integração Pesada* e *Crítico / Segurança*.

---

## 🚀 Funcionalidades

- **Painel Central (Dashboard):** Visualização e rastreamento de todos os requisitos funcionais submetidos.
- **Pipeline de IA Multidimensional:** Extração de regras, geração de BDD (Gherkin), tabela de testes e automação Playwright.
- **Aba "🤖 Insights de ML":** Na visualização de cada requisito, exibe em tempo real o esforço previsto, o risco de defeito, o perfil aglomerado, as recomendações técnicas do QA e a tabela de features extraídas.
- **Laboratórios Interativos de Machine Learning:**
  - `/ml/regressao/`: Simulação de parâmetros, tabela comparativa de métricas ($R^2$, MAE, RMSE) e gráfico de importância de atributos.
  - `/ml/classificacao/`: Controle dinâmico do limiar de corte (*threshold*), matriz de confusão e curva ROC.
  - `/ml/clusterizacao/`: Projeção cartesiana PCA 2D destacando o requisito sob análise contra os centróides e clusters históricos.
- **Jupyter Notebook Acadêmico:** Estudo experimental completo e reprodutível em `notebooks/AM_QA_Gen_Insights.ipynb`.

---

## 🛠️ Tecnologias Utilizadas

- **Backend:** Python 3.9+, Django 4.2+
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`
- **Frontend:** HTML5, CSS3, JavaScript & Bootstrap 5 (`django-bootstrap5`)
- **Persistência:** SQLite3 e serialização binária com Joblib (`ml_models/`)
- **Ambiente Interativo:** Jupyter Notebook (`nbconvert`, `ipykernel`)

---

## ⚙️ Instalação e Execução

### 1. Clonar ou Acessar a Pasta do Projeto
```bash
# Caso esteja clonando o repositório:
git clone <url-do-repositorio>
cd AM_Trabalho

# Ou apenas navegue até a pasta raiz do projeto no seu terminal
```

### 2. Criar e Ativar o Ambiente Virtual
```bash
# Criar o ambiente virtual (caso ainda não exista)
python3 -m venv portal-env

# Ativar no macOS / Linux:
source portal-env/bin/activate

# Ativar no Windows:
# portal-env\Scripts\activate
```

### 3. Instalar Dependências (caso necessário)
```bash
pip install -r requirements.txt
```

### 4. Gerar o Dataset e Treinar os Modelos de ML
O sistema já conta com o dataset e os modelos pré-treinados, mas podem ser reexecutados a qualquer momento:
```bash
# Gerar base sintética calibrada com 400 requisitos
python scripts/gerar_dataset.py

# Treinar os pipelines e salvar em ml_models/
python manage.py treinar_modelos
```

### 5. Executar o Servidor Web
```bash
python manage.py runserver
```
Acesse no navegador: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🧪 Testes Automatizados

Para executar a suíte completa de testes unitários e de integração de Machine Learning:
```bash
python manage.py test portal -v2
```

---

## 📓 Jupyter Notebook Didático

Para visualizar ou reexecutar o notebook da disciplina com todos os gráficos e análises teóricas:
```bash
# Abrir o ambiente interativo
jupyter notebook notebooks/AM_QA_Gen_Insights.ipynb

# Ou reexecutar em lote via linha de comando
jupyter nbconvert --to notebook --execute notebooks/AM_QA_Gen_Insights.ipynb --inplace
```

---

## 📂 Estrutura de Diretórios

```
AM_Trabalho/
├── data/
│   └── historico_requisitos.csv      # Dataset simulado com 400 registros
├── docs/
│   └── proposta_trabalho_am.md       # Proposta acadêmica da disciplina
├── ml_models/
│   ├── regressao.joblib              # Pipeline RandomForest treinado
│   ├── regressao_linear.joblib       # Pipeline LinearRegression treinado
│   ├── classificacao.joblib          # Pipeline Regressão Logística treinado
│   ├── classificacao_arvore.joblib   # Pipeline Árvore de Decisão treinado
│   ├── clusterizacao.joblib          # Artefatos KMeans + PCA + Centróides
│   └── metricas.json                 # Métricas consolidadas de validação cruzada
├── notebooks/
│   └── AM_QA_Gen_Insights.ipynb      # Notebook acadêmico com experimentos e gráficos
├── portal/
│   ├── ml/
│   │   └── features.py               # Extrator automatizado de features por regex
│   ├── ml_services.py                # Motor central de AM (treino, predição, plots)
│   ├── management/commands/
│   │   └── treinar_modelos.py        # Comando 'python manage.py treinar_modelos'
│   ├── models.py                     # Modelo Requisito
│   ├── views.py                      # Views do Django com suporte a insights de ML
│   ├── templates/                    # Templates HTML atualizados
│   └── tests.py                      # Suíte de testes automatizados
├── scripts/
│   ├── gerar_dataset.py              # Gerador do dataset com semente fixa
│   └── criar_notebook.py             # Script construtor do Jupyter Notebook
├── requirements.txt                  # Lista de dependências
└── manage.py                         # CLI do Django
```
