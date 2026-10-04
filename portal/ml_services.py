"""
Motor de Machine Learning (ml_services.py)
Módulo responsável pela lógica de Regressão, Classificação e Clusterização
para o QA Gen Web Portal.

Utiliza scikit-learn para os modelos e numpy para geração de datasets sintéticos.
Os modelos são treinados na primeira requisição (lazy loading) e cacheados em memória.
"""
import io
import base64
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para gerar imagens sem display
import matplotlib.pyplot as plt


# =====================================================================
# Cache global dos modelos (evita re-treino a cada request)
# =====================================================================
_cache_regressao = None
_cache_classificacao = None


# =====================================================================
# 1. REGRESSÃO — Prever Esforço de Teste (horas)
# =====================================================================

FEATURES_REGRESSAO = ['complexidade', 'num_integracoes', 'tamanho_requisito', 'num_regras_negocio']


def gerar_dataset_regressao(n_amostras=500, seed=42):
    """
    Gera um dataset sintético para treino do modelo de regressão.
    Simula a relação: esforço_horas ≈ f(complexidade, integrações, tamanho, regras).
    """
    rng = np.random.RandomState(seed)

    complexidade = rng.randint(1, 11, size=n_amostras)
    num_integracoes = rng.randint(0, 11, size=n_amostras)
    tamanho_requisito = rng.randint(50, 2001, size=n_amostras)
    num_regras_negocio = rng.randint(0, 21, size=n_amostras)

    # Fórmula sintética com ruído: esforço depende principalmente da complexidade
    # e do número de regras, com contribuições menores das outras features.
    esforco_horas = (
        complexidade * 1.5
        + num_integracoes * 0.8
        + tamanho_requisito * 0.005
        + num_regras_negocio * 0.6
        + rng.normal(0, 1.5, size=n_amostras)
    ).clip(min=0.5)  # Mínimo de 0.5 horas

    X = np.column_stack([complexidade, num_integracoes, tamanho_requisito, num_regras_negocio])
    y = esforco_horas

    return X, y


def treinar_modelo_regressao():
    """
    Treina o modelo de regressão e retorna modelo + métricas.
    Utiliza cache global para evitar re-treino.
    """
    global _cache_regressao

    if _cache_regressao is not None:
        return _cache_regressao

    X, y = gerar_dataset_regressao()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    metricas = {
        'r2': round(r2_score(y_test, y_pred), 4),
        'mae': round(mean_absolute_error(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
    }

    _cache_regressao = {
        'modelo': modelo,
        'metricas': metricas,
    }

    return _cache_regressao


def prever_esforco(complexidade, num_integracoes, tamanho_requisito, num_regras_negocio):
    """
    Faz a previsão de esforço de teste em horas para um novo requisito.
    Retorna a previsão e as métricas do modelo.
    """
    cache = treinar_modelo_regressao()
    modelo = cache['modelo']
    metricas = cache['metricas']

    X_input = np.array([[complexidade, num_integracoes, tamanho_requisito, num_regras_negocio]])
    previsao = modelo.predict(X_input)[0]

    return {
        'previsao': round(float(previsao), 2),
        'metricas': metricas,
        'importancias': dict(zip(FEATURES_REGRESSAO, [round(float(v), 4) for v in modelo.feature_importances_])),
    }


def gerar_grafico_importancia():
    """
    Gera um gráfico de barras da importância das features do modelo de regressão.
    Retorna a imagem como string base64 para exibição inline no HTML.
    """
    cache = treinar_modelo_regressao()
    modelo = cache['modelo']

    importancias = modelo.feature_importances_
    labels = ['Complexidade', 'Nº Integrações', 'Tamanho Requisito', 'Nº Regras Negócio']
    cores = ['#0d6efd', '#198754', '#ffc107', '#dc3545']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, importancias, color=cores, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Importância Relativa', fontsize=11)
    ax.set_title('Importância das Features — RandomForestRegressor', fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(importancias) * 1.15)

    # Adicionar valores nas barras
    for bar, val in zip(bars, importancias):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Converter para base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64


# =====================================================================
# 2. CLASSIFICAÇÃO — Classificar Prioridade de Bug/Requisito
# =====================================================================

FEATURES_CLASSIFICACAO = [
    'impacto_usuario', 'complexidade_tecnica', 'frequencia_uso',
    'tem_dependencia_externa', 'risco_seguranca',
]

CLASSES_PRIORIDADE = ['Baixa', 'Média', 'Alta', 'Crítica']


def gerar_dataset_classificacao(n_amostras=600, seed=42):
    """
    Gera um dataset sintético para treino do modelo de classificação.
    Atribui prioridade com base em uma pontuação composta das features.
    """
    rng = np.random.RandomState(seed)

    impacto_usuario = rng.randint(1, 11, size=n_amostras)
    complexidade_tecnica = rng.randint(1, 11, size=n_amostras)
    frequencia_uso = rng.randint(1, 11, size=n_amostras)
    tem_dependencia_externa = rng.randint(0, 2, size=n_amostras)
    risco_seguranca = rng.randint(1, 11, size=n_amostras)

    # Pontuação composta para determinar a classe
    score = (
        impacto_usuario * 0.30
        + complexidade_tecnica * 0.15
        + frequencia_uso * 0.20
        + tem_dependencia_externa * 2.0
        + risco_seguranca * 0.35
        + rng.normal(0, 0.8, size=n_amostras)
    )

    # Mapear score para classes usando percentis
    labels = np.empty(n_amostras, dtype=object)
    p25 = np.percentile(score, 25)
    p50 = np.percentile(score, 50)
    p75 = np.percentile(score, 75)
    labels[score < p25] = 'Baixa'
    labels[(score >= p25) & (score < p50)] = 'Média'
    labels[(score >= p50) & (score < p75)] = 'Alta'
    labels[score >= p75] = 'Crítica'

    X = np.column_stack([
        impacto_usuario, complexidade_tecnica, frequencia_uso,
        tem_dependencia_externa, risco_seguranca,
    ])

    return X, labels


def treinar_modelo_classificacao():
    """
    Treina o modelo de classificação e retorna modelo + métricas + label encoder.
    Utiliza cache global para evitar re-treino.
    """
    global _cache_classificacao

    if _cache_classificacao is not None:
        return _cache_classificacao

    X, y = gerar_dataset_classificacao()

    le = LabelEncoder()
    le.fit(CLASSES_PRIORIDADE)
    y_encoded = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    modelo = GradientBoostingClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    metricas = {
        'acuracia': round(accuracy_score(y_test, y_pred), 4),
        'f1_macro': round(f1_score(y_test, y_pred, average='macro'), 4),
    }

    _cache_classificacao = {
        'modelo': modelo,
        'label_encoder': le,
        'metricas': metricas,
    }

    return _cache_classificacao


def classificar_prioridade(impacto_usuario, complexidade_tecnica, frequencia_uso,
                           tem_dependencia_externa, risco_seguranca):
    """
    Classifica a prioridade de um bug/requisito com base nas features informadas.
    Retorna a classe predita, as probabilidades e as métricas do modelo.
    """
    cache = treinar_modelo_classificacao()
    modelo = cache['modelo']
    le = cache['label_encoder']
    metricas = cache['metricas']

    X_input = np.array([[
        impacto_usuario, complexidade_tecnica, frequencia_uso,
        tem_dependencia_externa, risco_seguranca,
    ]])

    pred_encoded = modelo.predict(X_input)[0]
    prioridade = le.inverse_transform([pred_encoded])[0]

    probas = modelo.predict_proba(X_input)[0]
    # modelo.classes_ contém os índices numéricos (inteiros).
    # Mapear cada índice de volta para o nome legível via label encoder.
    probabilidades_nomes = {}
    for idx, prob in zip(modelo.classes_, probas):
        nome = le.inverse_transform([idx])[0]
        probabilidades_nomes[nome] = round(float(prob) * 100, 1)

    return {
        'prioridade': prioridade,
        'probabilidades': probabilidades_nomes,
        'metricas': metricas,
    }


# =====================================================================
# 3. CLUSTERIZAÇÃO — Agrupar Requisitos Similares
# =====================================================================

def extrair_features_requisito(requisito):
    """
    Extrai features numéricas de um objeto Requisito do banco de dados.
    Retorna um dicionário com as features extraídas.
    """
    descricao = requisito.descricao or ''

    return {
        'tamanho_descricao': len(descricao),
        'num_palavras': len(descricao.split()),
        'tem_interpretacao': 1 if requisito.interpretacao else 0,
        'tem_bdd': 1 if requisito.bdd else 0,
        'tem_playwright': 1 if requisito.playwright else 0,
        'tem_edge_cases': 1 if requisito.edge_cases else 0,
        'processado': 1 if requisito.processado else 0,
    }


def gerar_dataset_clusterizacao_sintetico(n_amostras=30, seed=42):
    """
    Gera um dataset sintético para clusterização quando não há
    requisitos suficientes no banco de dados.
    """
    rng = np.random.RandomState(seed)

    dados = []
    titulos = []
    for i in range(n_amostras):
        tamanho = rng.randint(50, 800)
        n_palavras = int(tamanho / rng.uniform(4, 7))
        processado = rng.choice([0, 1], p=[0.3, 0.7])

        dados.append({
            'tamanho_descricao': tamanho,
            'num_palavras': n_palavras,
            'tem_interpretacao': processado,
            'tem_bdd': processado,
            'tem_playwright': processado * rng.choice([0, 1], p=[0.2, 0.8]),
            'tem_edge_cases': processado * rng.choice([0, 1], p=[0.3, 0.7]),
            'processado': processado,
        })
        titulos.append(f'Requisito Sintético #{i + 1}')

    return dados, titulos


def executar_clusterizacao(dados_features, titulos, n_clusters=3):
    """
    Executa a clusterização KMeans sobre os dados fornecidos.
    Retorna os clusters atribuídos, centróides e métricas.

    Args:
        dados_features: lista de dicts com features numéricas
        titulos: lista de títulos dos requisitos
        n_clusters: número de clusters desejados
    """
    feature_names = [
        'tamanho_descricao', 'num_palavras', 'tem_interpretacao',
        'tem_bdd', 'tem_playwright', 'tem_edge_cases', 'processado',
    ]

    # Montar matriz de features
    X = np.array([
        [d[f] for f in feature_names]
        for d in dados_features
    ], dtype=float)

    # Normalizar features para o KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ajustar número de clusters se necessário
    n_clusters = min(n_clusters, len(X))
    if n_clusters < 2:
        n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calcular silhouette score (precisa de pelo menos 2 clusters e mais amostras que clusters)
    if len(set(labels)) >= 2 and len(X) > n_clusters:
        sil_score = round(float(silhouette_score(X_scaled, labels)), 4)
    else:
        sil_score = None

    # Montar resultado por requisito
    resultados = []
    for i, (titulo, cluster) in enumerate(zip(titulos, labels)):
        resultados.append({
            'titulo': titulo,
            'cluster': int(cluster),
            'features': dados_features[i],
        })

    # Centróides (desnormalizados para interpretabilidade)
    centroides_reais = scaler.inverse_transform(kmeans.cluster_centers_)
    centroides = []
    for c_idx, centroide in enumerate(centroides_reais):
        centroides.append({
            'cluster': c_idx,
            'valores': {
                nome: round(float(val), 2)
                for nome, val in zip(feature_names, centroide)
            }
        })

    return {
        'resultados': resultados,
        'centroides': centroides,
        'inercia': round(float(kmeans.inertia_), 2),
        'silhouette': sil_score,
        'n_clusters': n_clusters,
    }
