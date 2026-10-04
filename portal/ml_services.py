"""
Motor de Machine Learning (ml_services.py) - QA Gen Insights
Implementa os 3 paradigmas da disciplina de Aprendizado de Máquina:
1. Regressão (Supervisionado Contínuo): LinearRegression vs RandomForestRegressor (K-Fold CV, R², MAE, RMSE)
2. Classificação (Supervisionado Discreto): LogisticRegression vs DecisionTreeClassifier (Stratified K-Fold, Precision, Recall, F1, ROC-AUC, Limiar Customizado)
3. Clusterização (Não Supervisionado): KMeans + Método do Cotovelo + Silhueta + PCA 2D

Integra persistência via joblib e fallback de auto-treinamento.
"""
import os
import io
import json
import base64
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
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
MODELS_DIR = os.path.join(BASE_DIR, 'ml_models')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'historico_requisitos.csv')

# Cache em memória
_MODELS_CACHE = None


def carregar_dataset():
    """Carrega o histórico de requisitos simulado. Se não existir, gera na hora."""
    if not os.path.exists(DATA_PATH):
        from scripts.gerar_dataset import gerar_historico_requisitos
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df = gerar_historico_requisitos()
        df.to_csv(DATA_PATH, index=False, encoding='utf-8')
    else:
        df = pd.read_csv(DATA_PATH)
    return df


def rotular_clusters(kmeans_model, scaler, feature_names):
    """
    Atribui nomes semânticos aos clusters a partir dos centróides desnormalizados.
    Identifica CRUD Simples, Fluxo Complexo, Integração Pesada e Crítico / Segurança.
    """
    centroides = scaler.inverse_transform(kmeans_model.cluster_centers_)
    nomes_clusters = {}
    
    # Índices das features
    idx_regras = feature_names.index('num_regras_negocio')
    idx_integracoes = feature_names.index('num_integracoes')
    idx_seguranca = feature_names.index('envolve_seguranca')
    
    for c_idx, centroide in enumerate(centroides):
        seg = centroide[idx_seguranca]
        integ = centroide[idx_integracoes]
        regras = centroide[idx_regras]
        
        if seg >= 0.5:
            nome = 'Crítico / Segurança'
            descricao = 'Foco em autenticação, proteção de dados e cenários sensíveis.'
            cor = 'danger'
        elif integ >= 2.5:
            nome = 'Integração Pesada'
            descricao = 'Comunicação com APIs, mensageria e serviços de terceiros.'
            cor = 'warning'
        elif regras >= 7.0:
            nome = 'Fluxo de Negócio Complexo'
            descricao = 'Múltiplas regras, ramificações e alta densidade de casos de teste.'
            cor = 'primary'
        else:
            nome = 'CRUD Simples'
            descricao = 'Operações básicas com baixo acoplamento e esforço enxuto.'
            cor = 'success'
            
        nomes_clusters[c_idx] = {
            'nome': nome,
            'descricao': descricao,
            'cor': cor,
            'centroide': {k: round(float(v), 2) for k, v in zip(feature_names, centroide)}
        }
    return nomes_clusters


def treinar_e_salvar_modelos():
    """
    Treina todos os modelos, computa métricas formais das aulas e persiste em disco.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = carregar_dataset()
    
    X = df[FEATURE_NAMES].values
    y_reg = df['esforco_teste_horas'].values
    y_clf = df['gerou_bug_producao'].values
    
    # -----------------------------------------------------------------
    # 1. REGRESSÃO: LinearRegression vs RandomForestRegressor (5-Fold CV)
    # -----------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Pipeline Linear
    pipe_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    # Pipeline Random Forest
    pipe_rf = Pipeline([
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    scoring_reg = {
        'r2': 'r2',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_rmse': 'neg_root_mean_squared_error',
    }
    
    cv_linear = cross_validate(pipe_linear, X, y_reg, cv=kf, scoring=scoring_reg)
    cv_rf = cross_validate(pipe_rf, X, y_reg, cv=kf, scoring=scoring_reg)
    
    # Ajuste final nos dados completos
    pipe_linear.fit(X, y_reg)
    pipe_rf.fit(X, y_reg)
    
    metricas_reg = {
        'linear': {
            'r2': round(float(np.mean(cv_linear['test_r2'])), 4),
            'mae': round(float(-np.mean(cv_linear['test_neg_mae'])), 4),
            'rmse': round(float(-np.mean(cv_linear['test_neg_rmse'])), 4),
            'coeficientes': {
                feat: round(float(coef), 4)
                for feat, coef in zip(FEATURE_NAMES, pipe_linear.named_steps['regressor'].coef_)
            },
            'intercept': round(float(pipe_linear.named_steps['regressor'].intercept_), 4)
        },
        'rf': {
            'r2': round(float(np.mean(cv_rf['test_r2'])), 4),
            'mae': round(float(-np.mean(cv_rf['test_neg_mae'])), 4),
            'rmse': round(float(-np.mean(cv_rf['test_neg_rmse'])), 4),
            'importancias': {
                feat: round(float(imp), 4)
                for feat, imp in zip(FEATURE_NAMES, pipe_rf.named_steps['regressor'].feature_importances_)
            }
        },
        'modelo_escolhido': 'RandomForestRegressor'
    }
    
    # -----------------------------------------------------------------
    # 2. CLASSIFICAÇÃO: LogisticRegression vs DecisionTree (Stratified 5-Fold)
    # -----------------------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    pipe_log = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    pipe_tree = Pipeline([
        ('classifier', DecisionTreeClassifier(max_depth=4, random_state=42))
    ])
    
    scoring_clf = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }
    
    cv_log = cross_validate(pipe_log, X, y_clf, cv=skf, scoring=scoring_clf)
    cv_tree = cross_validate(pipe_tree, X, y_clf, cv=skf, scoring=scoring_clf)
    
    # Ajuste final
    pipe_log.fit(X, y_clf)
    pipe_tree.fit(X, y_clf)
    
    # Calcular matriz de confusão e ROC Curve para exibição
    y_pred_log = pipe_log.predict(X)
    y_prob_log = pipe_log.predict_proba(X)[:, 1]
    cm = confusion_matrix(y_clf, y_pred_log).tolist()
    
    metricas_clf = {
        'logistica': {
            'accuracy': round(float(np.mean(cv_log['test_accuracy'])), 4),
            'precision': round(float(np.mean(cv_log['test_precision'])), 4),
            'recall': round(float(np.mean(cv_log['test_recall'])), 4),
            'f1': round(float(np.mean(cv_log['test_f1'])), 4),
            'roc_auc': round(float(np.mean(cv_log['test_roc_auc'])), 4),
            'matriz_confusao': cm,
        },
        'arvore': {
            'accuracy': round(float(np.mean(cv_tree['test_accuracy'])), 4),
            'precision': round(float(np.mean(cv_tree['test_precision'])), 4),
            'recall': round(float(np.mean(cv_tree['test_recall'])), 4),
            'f1': round(float(np.mean(cv_tree['test_f1'])), 4),
            'roc_auc': round(float(np.mean(cv_tree['test_roc_auc'])), 4),
        },
        'taxa_positivos': round(float(np.mean(y_clf)), 4),
        'limiar_padrao': 0.35,
        'modelo_escolhido': 'Regressão Logística (com limiar calibrado)'
    }
    
    # -----------------------------------------------------------------
    # 3. CLUSTERIZAÇÃO: K-Means (Cotovelo + Silhueta) + PCA 2D
    # -----------------------------------------------------------------
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X)
    
    cotovelo_dados = []
    silhueta_dados = []
    k_range = list(range(2, 9))
    
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_test = km_test.fit_predict(X_scaled)
        cotovelo_dados.append(round(float(km_test.inertia_), 2))
        silhueta_dados.append(round(float(silhouette_score(X_scaled, labels_test)), 4))
        
    # K ótimo = 4 (alinhado com os arquétipos e silhueta alta)
    k_otimo = 4
    kmeans_final = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X_scaled)
    
    # PCA para projeção 2D
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(X_scaled)
    variancia_explicada = [round(float(v), 4) for v in pca_2d.explained_variance_ratio_]
    
    nomes_clusters = rotular_clusters(kmeans_final, scaler_cluster, FEATURE_NAMES)
    
    cluster_artifacts = {
        'scaler': scaler_cluster,
        'kmeans': kmeans_final,
        'pca': pca_2d,
        'k_otimo': k_otimo,
        'nomes_clusters': nomes_clusters,
        'X_pca': X_pca.tolist(),
        'labels': labels_final.tolist(),
        'titulos': df['titulo'].tolist(),
    }
    
    metricas_clust = {
        'k_otimo': k_otimo,
        'inercia': round(float(kmeans_final.inertia_), 2),
        'silhouette': round(float(silhouette_score(X_scaled, labels_final)), 4),
        'variancia_explicada_pca': variancia_explicada,
        'variancia_total_retida': round(float(np.sum(variancia_explicada)), 4),
        'cotovelo_k': k_range,
        'cotovelo_inercia': cotovelo_dados,
        'silhueta_scores': silhueta_dados,
        'perfis': nomes_clusters,
    }
    
    # -----------------------------------------------------------------
    # Salvar em Disco
    # -----------------------------------------------------------------
    joblib.dump(pipe_rf, os.path.join(MODELS_DIR, 'regressao.joblib'))
    joblib.dump(pipe_linear, os.path.join(MODELS_DIR, 'regressao_linear.joblib'))
    joblib.dump(pipe_log, os.path.join(MODELS_DIR, 'classificacao.joblib'))
    joblib.dump(pipe_tree, os.path.join(MODELS_DIR, 'classificacao_arvore.joblib'))
    joblib.dump(cluster_artifacts, os.path.join(MODELS_DIR, 'clusterizacao.joblib'))
    
    todas_metricas = {
        'regressao': metricas_reg,
        'classificacao': metricas_clf,
        'clusterizacao': metricas_clust,
    }
    
    with open(os.path.join(MODELS_DIR, 'metricas.json'), 'w', encoding='utf-8') as f:
        json.dump(todas_metricas, f, indent=2, ensure_ascii=False)
        
    return todas_metricas


def carregar_modelos():
    """
    Obtém os modelos treinados do cache ou disco.
    Caso não existam arquivos, treina e inicializa transparentemente.
    """
    global _MODELS_CACHE
    
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE
        
    arq_reg = os.path.join(MODELS_DIR, 'regressao.joblib')
    arq_clf = os.path.join(MODELS_DIR, 'classificacao.joblib')
    arq_clust = os.path.join(MODELS_DIR, 'clusterizacao.joblib')
    arq_meta = os.path.join(MODELS_DIR, 'metricas.json')
    
    if not (os.path.exists(arq_reg) and os.path.exists(arq_clf) and os.path.exists(arq_clust) and os.path.exists(arq_meta)):
        treinar_e_salvar_modelos()
        
    with open(arq_meta, 'r', encoding='utf-8') as f:
        metricas = json.load(f)
        
    _MODELS_CACHE = {
        'regressao': joblib.load(arq_reg),
        'regressao_linear': joblib.load(os.path.join(MODELS_DIR, 'regressao_linear.joblib')),
        'classificacao': joblib.load(arq_clf),
        'classificacao_arvore': joblib.load(os.path.join(MODELS_DIR, 'classificacao_arvore.joblib')),
        'clusterizacao': joblib.load(arq_clust),
        'metricas': metricas,
    }
    return _MODELS_CACHE


# =====================================================================
# Funções de Predição e Avaliação
# =====================================================================

def prever_esforco(features_dict: dict) -> dict:
    """Preve esforço de teste (horas) usando o modelo treinado."""
    cache = carregar_modelos()
    modelo = cache['regressao']
    vetor = np.array([dict_para_vetor(features_dict)])
    pred = modelo.predict(vetor)[0]
    return {
        'esforco_horas': round(float(pred), 1),
        'metricas': cache['metricas']['regressao'],
    }


def prever_risco(features_dict: dict, limiar=0.35) -> dict:
    """
    Estima a probabilidade de bug em produção e aplica limiar de decisão.
    Limiar padrão 0.35 focado em Recall (prevenção de defeitos).
    """
    cache = carregar_modelos()
    modelo = cache['classificacao']
    vetor = np.array([dict_para_vetor(features_dict)])
    
    prob_bug = float(modelo.predict_proba(vetor)[0][1])
    class_pred = int(prob_bug >= limiar)
    
    if prob_bug < 0.25:
        nivel = 'Baixo'
        cor = 'success'
    elif prob_bug < 0.50:
        nivel = 'Médio'
        cor = 'warning'
    else:
        nivel = 'Alto'
        cor = 'danger'
        
    return {
        'probabilidade_bug': round(prob_bug * 100, 1),
        'prob_decimal': round(prob_bug, 4),
        'limiar_usado': round(float(limiar), 2),
        'gerou_alerta': bool(class_pred),
        'nivel_risco': nivel,
        'cor': cor,
        'metricas': cache['metricas']['classificacao'],
    }


def atribuir_perfil(features_dict: dict) -> dict:
    """Classifica o requisito em um cluster de perfil usando KMeans + PCA."""
    cache = carregar_modelos()
    clust_pkg = cache['clusterizacao']
    
    scaler = clust_pkg['scaler']
    kmeans = clust_pkg['kmeans']
    pca = clust_pkg['pca']
    nomes = clust_pkg['nomes_clusters']
    
    vetor = np.array([dict_para_vetor(features_dict)])
    vetor_scaled = scaler.transform(vetor)
    cluster_id = int(kmeans.predict(vetor_scaled)[0])
    coord_pca = pca.transform(vetor_scaled)[0]
    
    perfil_info = nomes.get(cluster_id, {
        'nome': f'Cluster #{cluster_id}',
        'descricao': 'Perfil agrupado via K-Means',
        'cor': 'info',
        'centroide': {}
    })
    
    return {
        'cluster_id': cluster_id,
        'nome': perfil_info['nome'],
        'descricao': perfil_info['descricao'],
        'cor': perfil_info['cor'],
        'coord_pca_2d': [round(float(coord_pca[0]), 3), round(float(coord_pca[1]), 3)],
        'centroide': perfil_info.get('centroide', {}),
        'metricas': cache['metricas']['clusterizacao'],
    }


def obter_insights_requisito(requisito, limiar=0.35) -> dict:
    """
    Função agregadora chamada na visualização do Requisito (resultado.html).
    Extrai features automaticamente e calcula Regressão, Classificação e Clusterização.
    """
    feats = extrair_features_requisito(requisito)
    res_reg = prever_esforco(feats)
    res_clf = prever_risco(feats, limiar=limiar)
    res_clust = atribuir_perfil(feats)
    
    # Gerar recomendação acionável para o QA
    recomendacoes = []
    if res_clf['nivel_risco'] in ['Médio', 'Alto']:
        if feats['num_edge_cases'] <= 3:
            recomendacoes.append("⚠️ Risco elevado detectado com baixa densidade de Edge Cases. Adicione cenários de falha de conexão e limites de payload.")
        if feats['envolve_seguranca']:
            recomendacoes.append("🔒 Requisito sensível de segurança: certifique-se de executar testes negativos de token inválido e rate-limiting.")
        if feats['num_integracoes'] >= 2:
            recomendacoes.append("🌐 Dependência de serviços externos: configure mocks e testes de timeout nos scripts Playwright.")
    else:
        recomendacoes.append("✅ Requisito com perfil controlado. Esforço enxuto com foco em testes de regressão do caminho feliz.")
        
    return {
        'features': feats,
        'regressao': res_reg,
        'classificacao': res_clf,
        'clusterizacao': res_clust,
        'recomendacoes': recomendacoes,
    }


# =====================================================================
# Gráficos em Base64 para visualização nas páginas Django
# =====================================================================

def gerar_grafico_regressao_importancia():
    """Gráfico de barras das importâncias e coeficientes de regressão."""
    cache = carregar_modelos()
    metricas = cache['metricas']['regressao']
    
    labels_map = {
        'num_regras_negocio': 'Regras Negócio',
        'num_cenarios_bdd': 'Cenários BDD',
        'num_casos_teste': 'Casos de Teste',
        'num_edge_cases': 'Edge Cases',
        'num_integracoes': 'Integrações',
        'envolve_seguranca': 'Segurança (0/1)',
        'tamanho_descricao': 'Palavras Descrição',
    }
    
    labels = [labels_map[k] for k in FEATURE_NAMES]
    importancias = [metricas['rf']['importancias'][k] for k in FEATURE_NAMES]
    
    fig, ax = plt.subplots(figsize=(7, 3.8), dpi=120)
    cores = ['#0d6efd', '#6610f2', '#6f42c1', '#d63384', '#fd7e14', '#198754', '#20c997']
    bars = ax.barh(labels, importancias, color=cores)
    ax.set_xlabel('Importância Relativa (Gini Importance)', fontsize=10)
    ax.set_title('RandomForestRegressor — Fatores Determinantes de Esforço', fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(importancias) * 1.15)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f'{w:.3f}', va='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def gerar_grafico_classificacao_roc():
    """Gera visualização da Curva ROC e Matriz de Confusão."""
    cache = carregar_modelos()
    pipe_log = cache['classificacao']
    metricas = cache['metricas']['classificacao']
    
    df = carregar_dataset()
    X = df[FEATURE_NAMES].values
    y = df['gerou_bug_producao'].values
    y_prob = pipe_log.predict_proba(X)[:, 1]
    
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc_val = metricas['logistica']['roc_auc']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8), dpi=120)
    
    # 1. Curva ROC
    ax1.plot(fpr, tpr, color='#0d6efd', lw=2, label=f'Regressão Logística (AUC = {auc_val:.3f})')
    ax1.plot([0, 1], [0, 1], color='#6c757d', linestyle='--', label='Aleatório (AUC = 0.500)')
    ax1.set_xlabel('Taxa de Falso Positivo (1 - Especificidade)', fontsize=9)
    ax1.set_ylabel('Taxa de Verdadeiro Positivo (Recall)', fontsize=9)
    ax1.set_title('Curva ROC — Detecção de Bugs', fontsize=10, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Matriz de Confusão Heatmap
    cm = np.array(metricas['logistica']['matriz_confusao'])
    im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax2.set_title('Matriz de Confusão (Dados Históricos)', fontsize=10, fontweight='bold')
    tick_marks = np.arange(2)
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(['Sem Defeito (0)', 'Com Bug (1)'], fontsize=9)
    ax2.set_yticklabels(['Sem Defeito (0)', 'Com Bug (1)'], fontsize=9)
    ax2.set_ylabel('Real', fontsize=9)
    ax2.set_xlabel('Previsto', fontsize=9)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontweight='bold', fontsize=11)
                     
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def gerar_grafico_clusterizacao_pca(destacar_ponto=None):
    """
    Gráfico de dispersão 2D via PCA com centróides e dispersão dos requisitos históricos.
    destacar_ponto: [coord_x, coord_y] do requisito atual sob análise.
    """
    cache = carregar_modelos()
    clust_pkg = cache['clusterizacao']
    
    X_pca = np.array(clust_pkg['X_pca'])
    labels = np.array(clust_pkg['labels'])
    nomes = clust_pkg['nomes_clusters']
    pca = clust_pkg['pca']
    kmeans = clust_pkg['kmeans']
    
    # Centróides projetados no PCA 2D
    centroides_pca = pca.transform(kmeans.cluster_centers_)
    
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=120)
    cores_paleta = {0: '#0d6efd', 1: '#198754', 2: '#ffc107', 3: '#dc3545'}
    
    for c_id in np.unique(labels):
        mask = labels == c_id
        nome_perfil = nomes[c_id]['nome']
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=f"{nome_perfil}",
            alpha=0.6,
            s=40,
            edgecolors='none'
        )
        
    # Plotar Centróides
    ax.scatter(
        centroides_pca[:, 0], centroides_pca[:, 1],
        marker='X', s=160, color='black',
        label='Centróides (K-Means)',
        zorder=5
    )
    
    if destacar_ponto is not None:
        ax.scatter(
            destacar_ponto[0], destacar_ponto[1],
            marker='*', s=280, color='#d63384',
            label='Requisito Atual (Sob Análise)',
            edgecolors='white', linewidths=1.5,
            zorder=10
        )
        
    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'Componente Principal 1 ({var_exp[0]*100:.1f}% variância)', fontsize=9)
    ax.set_ylabel(f'Componente Principal 2 ({var_exp[1]*100:.1f}% variância)', fontsize=9)
    ax.set_title('Projeção PCA 2D — Mapeamento de Requisitos por Perfil', fontsize=11, fontweight='bold')
    ax.legend(loc='best', fontsize=8, framealpha=0.85)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')
