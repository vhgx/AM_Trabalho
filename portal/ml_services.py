import os
import io
import base64
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from portal.ml.features import (
    FEATURE_NAMES,
    extrair_features_requisito,
    dict_para_vetor,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'historico_requisitos.csv')

_MODELS_CACHE = None

LABELS_FEATURES = {
    'num_regras_negocio': 'Regras de Negócio',
    'num_casos_teste': 'Casos de Teste',
    'num_integracoes': 'Integrações Externas',
    'envolve_seguranca': 'Envolve Segurança',
}


def gerar_dataset_padrao(n_amostras=300, seed=42) -> pd.DataFrame:
    """Gera base histórica sintética simples caso o CSV não exista."""
    rng = np.random.RandomState(seed)
    perfis = rng.choice(
        ['CRUD Simples', 'Fluxo de Negócio', 'Integração Externa', 'Crítico / Segurança'],
        size=n_amostras,
        p=[0.35, 0.30, 0.20, 0.15]
    )

    exemplos = {
        'CRUD Simples': [
            'Cadastro de categorias de produtos',
            'Listagem de usuários com filtro simples',
            'Edição de dados cadastrais do cliente',
            'Exclusão lógica de fornecedores inativos',
            'Consulta de tabela de preços vigentes',
        ],
        'Fluxo de Negócio': [
            'Cálculo de impostos interestaduais',
            'Aprovação de adiantamento financeiro',
            'Processamento de comissões de vendas',
            'Fechamento contábil mensal',
            'Renegociação de contratos parcelados',
        ],
        'Integração Externa': [
            'Webhook de notificações de pagamento',
            'Sincronização de clientes com CRM externo',
            'Emissão de nota fiscal via webservice',
            'Integração com gateway de pagamento PIX',
            'Consulta de frete em API de transportadora',
        ],
        'Crítico / Segurança': [
            'Autenticação em duas etapas (MFA)',
            'Anonimização de dados pessoais (LGPD)',
            'Validação de tokens JWT e controle de sessão',
            'Auditoria de acessos administrativos',
            'Recuperação de senha com token temporário',
        ]
    }

    registros = []
    for i, perfil in enumerate(perfis):
        titulo = rng.choice(exemplos[perfil])

        if perfil == 'CRUD Simples':
            regras = rng.randint(1, 5)
            casos = rng.randint(2, 7)
            integracoes = rng.choice([0, 1], p=[0.85, 0.15])
            seguranca = 0
        elif perfil == 'Fluxo de Negócio':
            regras = rng.randint(6, 15)
            casos = rng.randint(10, 22)
            integracoes = rng.randint(0, 3)
            seguranca = rng.choice([0, 1], p=[0.85, 0.15])
        elif perfil == 'Integração Externa':
            regras = rng.randint(3, 8)
            casos = rng.randint(6, 15)
            integracoes = rng.randint(3, 8)
            seguranca = rng.choice([0, 1], p=[0.7, 0.3])
        else:
            regras = rng.randint(4, 10)
            casos = rng.randint(8, 18)
            integracoes = rng.randint(1, 4)
            seguranca = 1

        esforco = (
            1.5
            + 0.65 * regras
            + 0.30 * casos
            + 0.90 * integracoes
            + 2.20 * seguranca
            + rng.normal(0, 0.8)
        )
        esforco = max(1.0, round(float(esforco), 1))

        logit = (
            -2.8
            + 0.18 * regras
            + 0.45 * integracoes
            + 1.30 * seguranca
            + 0.04 * casos
            + rng.normal(0, 0.5)
        )
        prob_bug = 1.0 / (1.0 + np.exp(-logit))
        gerou_bug = int(rng.uniform(0, 1) < prob_bug)

        registros.append({
            'id': i + 1,
            'titulo': titulo,
            'perfil': perfil,
            'num_regras_negocio': regras,
            'num_casos_teste': casos,
            'num_integracoes': integracoes,
            'envolve_seguranca': seguranca,
            'esforco_teste_horas': esforco,
            'gerou_bug_producao': gerou_bug,
        })

    return pd.DataFrame(registros)


def carregar_dataset() -> pd.DataFrame:
    """Lê o CSV de histórico de requisitos ou cria um novo se não existir."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = gerar_dataset_padrao()
        df.to_csv(DATA_PATH, index=False, encoding='utf-8')
        return df

    df = pd.read_csv(DATA_PATH)
    if not all(col in df.columns for col in FEATURE_NAMES):
        df = gerar_dataset_padrao()
        df.to_csv(DATA_PATH, index=False, encoding='utf-8')
    return df


def rotular_clusters(kmeans_model, scaler, feature_names):
    """Dá um nome descritivo para cada grupo do K-Means com base nas médias."""
    centroides = scaler.inverse_transform(kmeans_model.cluster_centers_)
    nomes_clusters = {}

    idx_regras = feature_names.index('num_regras_negocio')
    idx_integracoes = feature_names.index('num_integracoes')
    idx_seguranca = feature_names.index('envolve_seguranca')

    for c_idx, centroide in enumerate(centroides):
        seg = centroide[idx_seguranca]
        integ = centroide[idx_integracoes]
        regras = centroide[idx_regras]

        if seg >= 0.5:
            nome = 'Crítico / Segurança'
            descricao = 'Requisitos com autenticação, permissões ou dados sensíveis.'
            cor = 'danger'
        elif integ >= 2.5:
            nome = 'Integração Externa'
            descricao = 'Requisitos que dependem de APIs, webhooks ou serviços externos.'
            cor = 'warning'
        elif regras >= 6.5:
            nome = 'Fluxo de Negócio'
            descricao = 'Requisitos com várias regras de negócio e cenários de validação.'
            cor = 'primary'
        else:
            nome = 'CRUD Simples'
            descricao = 'Operações diretas de cadastro, listagem ou consulta.'
            cor = 'success'

        nomes_clusters[c_idx] = {
            'nome': nome,
            'descricao': descricao,
            'cor': cor,
            'centroide': {k: round(abs(float(v)) if abs(v) < 1e-6 else float(v), 1) for k, v in zip(feature_names, centroide)},
        }
    return nomes_clusters


def carregar_modelos():
    """Treina os 3 modelos em memória (ou retorna do cache)."""
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE

    df = carregar_dataset()
    X = df[FEATURE_NAMES].values
    y_reg = df['esforco_teste_horas'].values
    y_clf = df['gerou_bug_producao'].values

    # 1. Regressão (Random Forest)
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    modelo_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo_reg.fit(X_tr_r, y_tr_r)
    pred_reg = modelo_reg.predict(X_te_r)

    metricas_reg = {
        'r2': round(float(r2_score(y_te_r, pred_reg)), 2),
        'mae': round(float(mean_absolute_error(y_te_r, pred_reg)), 1),
        'rmse': round(float(np.sqrt(mean_squared_error(y_te_r, pred_reg))), 1),
        'importancias': {
            feat: round(float(imp), 3)
            for feat, imp in zip(FEATURE_NAMES, modelo_reg.feature_importances_)
        },
    }

    # 2. Classificação (Regressão Logística)
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    modelo_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42))
    ])
    modelo_clf.fit(X_tr_c, y_tr_c)
    pred_clf = modelo_clf.predict(X_te_c)

    metricas_clf = {
        'accuracy': round(float(accuracy_score(y_te_c, pred_clf)) * 100, 1),
        'precision': round(float(precision_score(y_te_c, pred_clf, zero_division=0)) * 100, 1),
        'recall': round(float(recall_score(y_te_c, pred_clf, zero_division=0)) * 100, 1),
        'f1': round(float(f1_score(y_te_c, pred_clf, zero_division=0)) * 100, 1),
    }

    # 3. Clusterização (K-Means)
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    nomes_clusters = rotular_clusters(kmeans, scaler_cluster, FEATURE_NAMES)

    contagem = {}
    for c_id in range(4):
        contagem[c_id] = int(np.sum(labels == c_id))

    metricas_clust = {
        'n_clusters': 4,
        'inercia': round(float(kmeans.inertia_), 1),
        'silhouette': round(float(silhouette_score(X_scaled, labels)), 2),
        'perfis': nomes_clusters,
        'contagem': contagem,
    }

    _MODELS_CACHE = {
        'regressao': modelo_reg,
        'classificacao': modelo_clf,
        'clusterizacao': {
            'scaler': scaler_cluster,
            'kmeans': kmeans,
            'nomes_clusters': nomes_clusters,
            'X': X,
            'labels': labels,
        },
        'metricas': {
            'regressao': metricas_reg,
            'classificacao': metricas_clf,
            'clusterizacao': metricas_clust,
        },
    }
    return _MODELS_CACHE


def prever_esforco(features_dict: dict) -> dict:
    """Prevê o esforço de teste em horas."""
    cache = carregar_modelos()
    modelo = cache['regressao']
    vetor = np.array([dict_para_vetor(features_dict)])
    pred = modelo.predict(vetor)[0]
    return {
        'esforco_horas': round(float(pred), 1),
        'metricas': cache['metricas']['regressao'],
    }


def prever_risco(features_dict: dict) -> dict:
    """Calcula a probabilidade de bug e classifica o risco em Baixo, Médio ou Alto."""
    cache = carregar_modelos()
    modelo = cache['classificacao']
    vetor = np.array([dict_para_vetor(features_dict)])

    prob_bug = float(modelo.predict_proba(vetor)[0][1])

    if prob_bug < 0.30:
        nivel = 'Baixo'
        cor = 'success'
    elif prob_bug < 0.60:
        nivel = 'Médio'
        cor = 'warning'
    else:
        nivel = 'Alto'
        cor = 'danger'

    return {
        'probabilidade_bug': round(prob_bug * 100, 1),
        'nivel_risco': nivel,
        'cor': cor,
        'metricas': cache['metricas']['classificacao'],
    }


def atribuir_perfil(features_dict: dict) -> dict:
    """Identifica em qual grupo (cluster) o requisito se encaixa."""
    cache = carregar_modelos()
    clust_pkg = cache['clusterizacao']

    scaler = clust_pkg['scaler']
    kmeans = clust_pkg['kmeans']
    nomes = clust_pkg['nomes_clusters']

    vetor = np.array([dict_para_vetor(features_dict)])
    vetor_scaled = scaler.transform(vetor)
    cluster_id = int(kmeans.predict(vetor_scaled)[0])

    perfil_info = nomes.get(cluster_id, {
        'nome': f'Grupo {cluster_id}',
        'descricao': 'Grupo identificado pelo K-Means',
        'cor': 'info',
        'centroide': {},
    })

    return {
        'cluster_id': cluster_id,
        'nome': perfil_info['nome'],
        'descricao': perfil_info['descricao'],
        'cor': perfil_info['cor'],
        'centroide': perfil_info.get('centroide', {}),
        'metricas': cache['metricas']['clusterizacao'],
    }


def obter_insights_requisito(requisito) -> dict:
    """Extrai os indicadores de um requisito salvo e roda os 3 modelos."""
    feats = extrair_features_requisito(requisito)
    res_reg = prever_esforco(feats)
    res_clf = prever_risco(feats)
    res_clust = atribuir_perfil(feats)

    dicas = []
    if res_clf['nivel_risco'] in ['Médio', 'Alto']:
        if feats['envolve_seguranca']:
            dicas.append("Inclua testes de tentativa de acesso não autorizado e expiração de sessão/token.")
        if feats['num_integracoes'] >= 2:
            dicas.append("Considere simular indisponibilidade ou lentidão dos serviços externos.")
        if not dicas:
            dicas.append("Pela quantidade de regras, vale revisar os cenários alternativos antes da homologação.")
    else:
        dicas.append("Requisito de baixa complexidade. Foque nos fluxos principais de cadastro e validação básica.")

    return {
        'features': feats,
        'regressao': res_reg,
        'classificacao': res_clf,
        'clusterizacao': res_clust,
        'recomendacoes': dicas,
    }


def gerar_grafico_regressao_importancia():
    """Gráfico simples de barras horizontais mostrando o peso de cada atributo."""
    cache = carregar_modelos()
    importancias = cache['metricas']['regressao']['importancias']

    labels = [LABELS_FEATURES[k] for k in FEATURE_NAMES]
    valores = [importancias[k] * 100 for k in FEATURE_NAMES]

    fig, ax = plt.subplots(figsize=(6.5, 3.0), dpi=110)
    bars = ax.barh(labels, valores, color='#0d6efd', height=0.55)
    ax.set_xlabel('Peso na estimativa (%)', fontsize=9)
    ax.set_title('Peso de cada fator no cálculo de horas', fontsize=10, fontweight='bold')
    ax.set_xlim(0, max(valores) * 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f'{w:.1f}%', va='center', fontsize=8.5)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def gerar_grafico_clusterizacao(features_atuais=None):
    """Gráfico de dispersão simples: Regras de Negócio x Casos de Teste por grupo."""
    cache = carregar_modelos()
    clust_pkg = cache['clusterizacao']

    X = clust_pkg['X']
    labels = clust_pkg['labels']
    nomes = clust_pkg['nomes_clusters']

    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=110)
    cores_map = {'success': '#198754', 'primary': '#0d6efd', 'warning': '#fd7e14', 'danger': '#dc3545'}

    for c_id in np.unique(labels):
        mask = labels == c_id
        info = nomes[c_id]
        cor_hex = cores_map.get(info['cor'], '#6c757d')
        ax.scatter(
            X[mask, 0], X[mask, 1],
            label=info['nome'],
            color=cor_hex,
            alpha=0.65,
            s=35,
        )

    if features_atuais is not None:
        ax.scatter(
            features_atuais['num_regras_negocio'],
            features_atuais['num_casos_teste'],
            color='#212529',
            marker='X',
            s=150,
            label='Requisito analisado',
            zorder=10,
        )

    ax.set_xlabel('Regras de Negócio', fontsize=9)
    ax.set_ylabel('Casos de Teste', fontsize=9)
    ax.set_title('Distribuição dos Requisitos por Grupo (K-Means)', fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
