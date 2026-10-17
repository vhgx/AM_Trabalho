# QA Gen Portal

Plataforma web em Django para geração automatizada de casos de teste a partir de requisitos funcionais (via LLM) e estimativas de qualidade usando Aprendizado de Máquina (`scikit-learn`).

Projeto desenvolvido para a Pós-Graduação em Inteligência Artificial (INF/UFG).

---

## Funcionalidades

### 1. Geração de Artefatos de Teste (LLM)
Ao cadastrar e processar um requisito funcional, o sistema gera:
- Interpretação e regras de negócio
- Cenários BDD (Gherkin)
- Tabela de casos de teste manuais
- Script de automação E2E (Playwright)
- Cenários de borda (*edge cases*)

### 2. Estimativas com Machine Learning (`scikit-learn`)
A partir de 4 indicadores extraídos do requisito (`num_regras_negocio`, `num_casos_teste`, `num_integracoes`, `envolve_seguranca`), o portal aplica três tarefas de Aprendizado de Máquina:

1. **Previsão de Esforço — Regressão (`/ml/regressao/`):**
   - Modelo: `RandomForestRegressor`
   - Estima o tempo de teste em horas e exibe $R^2$, erro médio (MAE) e o peso de cada indicador.
2. **Risco de Defeito — Classificação (`/ml/classificacao/`):**
   - Modelo: `LogisticRegression`
   - Classifica o risco de falha em produção em *Baixo*, *Médio* ou *Alto*, exibindo acurácia, precisão, recall e F1-score.
3. **Agrupamento de Perfis — Clusterização (`/ml/clusterizacao/`):**
   - Modelo: `KMeans` ($k=4$) com `StandardScaler`
   - Agrupa os requisitos sem rótulos prévios em quatro perfis (*CRUD Simples*, *Fluxo de Negócio*, *Integração Externa* e *Crítico / Segurança*).

Além das páginas de simulação no menu **Machine Learning**, a tela de resultado de cada requisito possui a aba **Estimativas (ML)** com o cálculo automático para o requisito visualizado.

---

## Como Executar

### 1. Ambiente virtual e dependências
```bash
python3 -m venv portal-env
source portal-env/bin/activate
pip install -r requirements.txt
```

### 2. Banco de dados e servidor
```bash
python manage.py migrate
python manage.py runserver
```
Acesse em `http://127.0.0.1:8000`.

*(Os modelos de Machine Learning são treinados automaticamente em memória a partir da base `data/historico_requisitos.csv` ao iniciar a aplicação.)*

### 3. Testes automatizados
```bash
python manage.py test portal -v2
```

---

## Estrutura Principal

```
AM_Trabalho/
├── data/
│   └── historico_requisitos.csv      # Base histórica de requisitos (300 registros)
├── docs/
│   └── proposta_trabalho_am.md       # Descrição do trabalho da disciplina
├── notebooks/
│   └── AM_QA_Gen_Insights.ipynb      # Notebook com análise exploratória e treino dos modelos
├── portal/
│   ├── ml/features.py                # Extração dos 4 indicadores a partir do texto do requisito
│   ├── ml_services.py                # Treino e predição (Regressão, Classificação e Clusterização)
│   ├── models.py                     # Modelo Requisito
│   ├── services.py                   # Integração com a API de LLM
│   ├── views.py                      # Views do portal e páginas de ML
│   ├── templates/                    # Templates HTML (Bootstrap 5)
│   └── tests.py                      # Testes automatizados
└── manage.py
```
