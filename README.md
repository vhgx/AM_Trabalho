# QA Gen Portal

Plataforma web em Django para geração automatizada de casos de teste a partir de requisitos funcionais (via LLM) e estimativas de qualidade usando Aprendizado de Máquina (`scikit-learn`).

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

### 2. Configuração do LLM (Local vs. API)
Copie o arquivo de exemplo de variáveis de ambiente:
```bash
cp .env.example .env
```
Por padrão, o arquivo vem preparado para **LM Studio local** (`http://localhost:1234/v1`). Você pode alterar as variáveis `OPENAI_BASE_URL`, `OPENAI_API_KEY` e `OPENAI_MODEL` para usar **Ollama**, **OpenAI**, **Groq**, **Gemini**, etc. Veja as opções na seção [Configuração do LLM (Local vs. API)](#configuracao-do-llm-local-vs-api).

### 3. Banco de dados e servidor
```bash
python manage.py migrate
python manage.py runserver
```
Acesse em `http://127.0.0.1:8000`.

*(Os modelos de Machine Learning são treinados automaticamente em memória a partir da base `data/historico_requisitos.csv` ao iniciar a aplicação.)*

### 4. Testes automatizados
```bash
python manage.py test portal -v2
```

---

## Configuração do LLM (Local vs. API)

O pipeline de geração de artefatos de teste (`portal/services.py`) utiliza a biblioteca oficial `openai`, sendo compatível com qualquer provedor ou servidor que implemente a especificação **OpenAI Chat Completions API**.

As chamadas são parametrizadas por três variáveis no arquivo `.env`:

| Variável | Descrição | Exemplo Local | Exemplo em Nuvem |
|---|---|---|---|
| `OPENAI_BASE_URL` | URL base do endpoint | `http://localhost:1234/v1` | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | Chave de autenticação | `lm-studio` ou `ollama` | `sk-proj-...` ou `gsk_...` |
| `OPENAI_MODEL` | Identificador do modelo | `local-model` ou `qwen2.5-coder:7b` | `gpt-4o-mini` ou `llama-3.3-70b-versatile` |

---

### Opção 1: Geração Local (100% Offline e Gratuita)

Recomendada para privacidade total dos dados do requisito, sem custos de API e com funcionamento sem conexão externa.

#### A. Usando o LM Studio (Interface Visual)
1. Instale o [LM Studio](https://lmstudio.ai/).
2. Na aba de busca, baixe um modelo instruído (sugestões: `Qwen2.5-Coder-7B-Instruct`, `Llama-3.1-8B-Instruct` ou `Mistral-7B-Instruct-v0.3`).
3. Acesse a aba **Developer / Local Server** (`<->` no menu lateral).
4. Selecione o modelo baixado no topo e clique em **Start Server** (porta padrão: `1234`).
5. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=http://localhost:1234/v1
   OPENAI_API_KEY=lm-studio
   OPENAI_MODEL=local-model
   ```

#### B. Usando o Ollama (Linha de Comando)
1. Instale o [Ollama](https://ollama.com/).
2. Execute o modelo desejado no terminal (o serviço compatível será iniciado na porta `11434`):
   ```bash
   ollama run qwen2.5-coder:7b
   ```
3. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   OPENAI_MODEL=qwen2.5-coder:7b
   ```

---

### Opção 2: Geração via API em Nuvem (Mais Rápido e Sem Exigência de GPU)

Recomendada se sua máquina não possuir GPU dedicada ou para obter respostas mais rápidas e com modelos de alta capacidade.

#### A. OpenAI (GPT-4o mini / GPT-4o)
1. Gere uma chave em [OpenAI Platform](https://platform.openai.com/api-keys).
2. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=https://api.openai.com/v1
   OPENAI_API_KEY=sk-proj-sua-chave-aqui
   OPENAI_MODEL=gpt-4o-mini
   ```

#### B. Groq (Altíssima Velocidade com Modelos Open-Source)
1. Crie uma chave gratuita no [Groq Console](https://console.groq.com/keys).
2. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=https://api.groq.com/openai/v1
   OPENAI_API_KEY=gsk_sua-chave-aqui
   OPENAI_MODEL=llama-3.3-70b-versatile
   ```

#### C. Google Gemini (Via Endpoint de Compatibilidade OpenAI)
1. Obtenha uma chave gratuita no [Google AI Studio](https://aistudio.google.com/).
2. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
   OPENAI_API_KEY=sua-chave-gemini-aqui
   OPENAI_MODEL=gemini-1.5-flash
   ```

#### D. OpenRouter
1. Crie uma chave em [OpenRouter](https://openrouter.ai/keys).
2. No arquivo `.env`:
   ```env
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_API_KEY=sk-or-sua-chave-aqui
   OPENAI_MODEL=meta-llama/llama-3.3-70b-instruct
   ```

---

### Comparativo Rápido

| Provedor | Modalidade | Custo | Exigência de Hardware | Latência / Velocidade |
|---|---|---|---|---|
| **LM Studio** | Local | Gratuito | Média/Alta (RAM / VRAM) | Depende da máquina |
| **Ollama** | Local | Gratuito | Média/Alta (RAM / VRAM) | Depende da máquina |
| **OpenAI (GPT-4o-mini)** | Nuvem (API) | Pago (centavos de dólar) | Nenhuma | Rápida (~1-3s) |
| **Groq (Llama 3.3)** | Nuvem (API) | Tier gratuito disponível | Nenhuma | Ultrarrápida (~500 tokens/s) |
| **Google Gemini (Flash)** | Nuvem (API) | Tier gratuito disponível | Nenhuma | Muito rápida (~1-2s) |

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
│   ├── services.py                   # Integração com a API de LLM (compatível com OpenAI API)
│   ├── views.py                      # Views do portal e páginas de ML
│   ├── templates/                    # Templates HTML (Bootstrap 5)
│   └── tests.py                      # Testes automatizados
├── .env.example                      # Template com exemplos de configuração de LLM (Local e Nuvem)
├── requirements.txt                  # Dependências do projeto
└── manage.py
```

