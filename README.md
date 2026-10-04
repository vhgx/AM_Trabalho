# 🧪 QA Gen Web Portal

**QA Gen Web Portal** é uma plataforma web desenvolvida em Django para a geração automatizada de artefatos de teste de software utilizando Inteligência Artificial (LLMs). 

Este projeto é a evolução do agente **ePrompt (QA Test Generator Agent)**, transformando uma ferramenta de linha de comando em um portal SaaS (Software as a Service) centralizado, capaz de persistir dados e exibir resultados de forma rica e organizada.

---

## 🚀 Funcionalidades

- **Painel Central (Dashboard):** Visualize e gerencie todos os requisitos submetidos e seus respectivos artefatos gerados.
- **Submissão de Contexto:** Interface amigável para descrever requisitos funcionais ou histórias de usuário.
- **Pipeline de IA Multidimensional:** O sistema processa o requisito em 5 etapas sequenciais:
  1. **Análise e Interpretação:** Extração de atores, pré-condições e regras de negócio.
  2. **Cenários BDD (Gherkin):** Escrita de funcionalidades seguindo o padrão Given/When/Then.
  3. **Casos de Teste Estruturados:** Tabela detalhada para execução de testes manuais.
  4. **Automação (Playwright):** Geração de scripts de teste E2E em JavaScript/Node.js.
  5. **Edge Cases:** Identificação de cenários extremos e pontos de falha de segurança.
- **Visualização Rica:** Resultados exibidos em abas com formatação de código (IDE-style) e tabelas renderizadas.
- **Machine Learning — Regressão:** Modelo `RandomForestRegressor` para prever o esforço de teste (em horas) com base em features do requisito. Exibe métricas (R², MAE, RMSE) e gráfico de importância das features.
- **Machine Learning — Classificação:** Modelo `GradientBoostingClassifier` para classificar a prioridade de bugs/requisitos em Crítica, Alta, Média ou Baixa. Exibe probabilidades por classe e métricas (Acurácia, F1-Score).
- **Machine Learning — Clusterização:** Modelo `KMeans` para agrupar requisitos similares sem rótulos prévios (Aprendizado Não Supervisionado). Exibe clusters, centróides e métricas (Inércia, Silhouette Score).

---

## 🛠️ Tecnologias Utilizadas

- **Backend:** [Python 3](https://www.python.org/) & [Django 4.2](https://www.djangoproject.com/)
- **Frontend:** HTML5, CSS3, JavaScript & [Bootstrap 5](https://getbootstrap.com/)
- **IA/LLM:** Integração via [OpenAI API](https://openai.com/blog/openai-api) (compatível com LM Studio, Gemini, Anthropic)
- **Banco de Dados:** SQLite (Relacional)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (RandomForest, GradientBoosting, KMeans), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/)
- **Estilização de Texto:** [Markdown](https://pypi.org/project/Markdown/) para renderização de tabelas e blocos de código.

---

## ⚙️ Instalação e Configuração

### 1. Clonar o Repositório
```bash
git clone https://github.com/vhgx/qa-gen-portal.git
cd qa-gen-portal
```

### 2. Configurar o Ambiente Virtual
```bash
# Criar o ambiente
python3 -m venv portal-env

# Ativar no Mac/Linux
source portal-env/bin/activate

# Ativar no Windows
# portal-env\Scripts\activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
# Ou manualmente:
pip install django django-bootstrap-v5 openai python-dotenv django-tables2 markdown scikit-learn numpy matplotlib
```

### 4. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto com as seguintes chaves:
```env
OPENAI_BASE_URL=http://localhost:1234/v1  # Ou URL da OpenAI/Gemini
OPENAI_API_KEY=lm-studio                  # Sua chave de API
OPENAI_MODEL=local-model                  # Nome do modelo
```

### 5. Migrar o Banco de Dados
```bash
python manage.py makemigrations
python manage.py migrate
```

### 6. Executar o Servidor
```bash
python manage.py runserver
```
Acesse em: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📂 Estrutura do Projeto

- `qagen/`: Configurações centrais do projeto Django.
- `portal/`: Aplicação principal contendo a lógica de negócio.
  - `services.py`: Motor de comunicação com a IA e orquestração do pipeline de QA.
  - `ml_services.py`: Motor de Machine Learning (Regressão, Classificação, Clusterização).
  - `models.py`: Estrutura de persistência dos requisitos e resultados.
  - `views.py`: Views do Django (Dashboard, Submissão, Resultado, ML).
  - `urls.py`: Rotas da aplicação.
  - `tests.py`: Testes automatizados (Django TestCase).
  - `templates/`: Camada de apresentação (HTML/Django Templates).
  - `static/`: Arquivos CSS e assets visuais.
  - `templatetags/`: Filtros customizados para renderização de Markdown.

---

## 🤖 Machine Learning

O portal inclui 3 funcionalidades de Machine Learning acessíveis via menu **"🤖 Machine Learning"** na navbar:

| Funcionalidade | URL | Modelo | Tipo de Aprendizado |
|---|---|---|---|
| **Regressão** — Prever Esforço de Teste | `/ml/regressao/` | `RandomForestRegressor` | Supervisionado (saída contínua) |
| **Classificação** — Prioridade de Requisito | `/ml/classificacao/` | `GradientBoostingClassifier` | Supervisionado (saída categórica) |
| **Clusterização** — Agrupar Requisitos | `/ml/clusterizacao/` | `KMeans` | Não Supervisionado |

Os modelos são treinados com datasets sintéticos gerados automaticamente (seed fixa para reprodutibilidade) e cacheados em memória após o primeiro uso.

### Executar Testes
```bash
python manage.py test portal -v2
```

---

## 🎓 Contexto Acadêmico
Este projeto foi desenvolvido como trabalho prático da disciplina de **Framework de Desenvolvimento Web** no curso de Pós-Graduação, integrando conhecimentos de engenharia de prompt, arquitetura MVC, consumo de APIs externas e Machine Learning (Regressão, Classificação e Clusterização).
