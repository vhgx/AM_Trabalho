from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Requisito
from .services import run_qa_pipeline
from .ml.features import FEATURE_NAMES, extrair_features_requisito
from .ml_services import (
    carregar_modelos,
    prever_esforco,
    prever_risco,
    atribuir_perfil,
    obter_insights_requisito,
    gerar_grafico_regressao_importancia,
    gerar_grafico_classificacao_roc,
    gerar_grafico_clusterizacao_pca,
)


def dashboard(request):
    """Painel Central (Dashboard) - exibe requisitos submetidos anteriormente."""
    requisitos = Requisito.objects.all()
    dicionario = {'requisitos': requisitos}
    return render(request, 'dashboard.html', dicionario)


def submeter(request):
    """Submissão de Contexto - recebe a história de usuário ou regra de negócio."""
    if request.method == 'POST':
        titulo = request.POST.get('titulo', '').strip()
        descricao = request.POST.get('descricao', '').strip()

        if not titulo or not descricao:
            dicionario = {
                'erro': 'Título e Requisito Funcional são campos obrigatórios.',
                'titulo': titulo,
                'descricao': descricao,
            }
            return render(request, 'submeter.html', dicionario)

        requisito = Requisito.objects.create(
            titulo=titulo,
            descricao=descricao,
            processado=False,
        )
        return redirect('processar', pk=requisito.pk)

    return render(request, 'submeter.html')


def processar(request, pk):
    """Executa o pipeline de QA via IA e redireciona para a tela de resultados."""
    requisito = get_object_or_404(Requisito, pk=pk)

    if not requisito.processado:
        try:
            resultados = run_qa_pipeline(requisito.descricao)
            requisito.interpretacao = resultados.get('interpretacao', '')
            requisito.bdd = resultados.get('bdd', '')
            requisito.casos_teste = resultados.get('casos_teste', '')
            requisito.playwright = resultados.get('playwright', '')
            requisito.edge_cases = resultados.get('edge_cases', '')
            requisito.processado = True
            requisito.save()
        except Exception as e:
            dicionario = {
                'erro': f'Falha na comunicação com o provedor de IA: {str(e)}',
                'requisito': requisito,
            }
            return render(request, 'erro.html', dicionario)

    return redirect('resultado', pk=requisito.pk)


def resultado(request, pk):
    """
    Exibição Rica e Multidimensional dos resultados gerados.
    Inclui os artefatos tradicionais de QA e a aba 'Insights de ML' calculada
    automaticamente a partir dos artefatos produzidos pelo LLM.
    """
    requisito = get_object_or_404(Requisito, pk=pk)
    insights = None
    
    if requisito.processado:
        try:
            insights = obter_insights_requisito(requisito)
        except Exception as e:
            insights = {'erro': f'Não foi possível computar os insights de ML: {str(e)}'}

    dicionario = {
        'requisito': requisito,
        'insights': insights,
    }
    return render(request, 'resultado.html', dicionario)


# =====================================================================
# Views de Machine Learning (Laboratórios Interativos)
# =====================================================================

def ml_regressao(request):
    """
    Laboratório de Regressão — Prever Esforço de Teste (em horas).
    Exibe comparativo formal entre Linear Regression e RandomForestRegressor (K-Fold CV, R², MAE, RMSE).
    """
    cache = carregar_modelos()
    metricas = cache['metricas']['regressao']
    
    # Valores padrão de entrada
    inputs = {
        'num_regras_negocio': 5,
        'num_cenarios_bdd': 3,
        'num_casos_teste': 8,
        'num_edge_cases': 3,
        'num_integracoes': 1,
        'envolve_seguranca': 0,
        'tamanho_descricao': 120,
    }
    resultado = None
    
    if request.method == 'POST':
        for campo in inputs.keys():
            val = request.POST.get(campo)
            if val is not None:
                inputs[campo] = int(val)
                
        resultado = prever_esforco(inputs)
    else:
        resultado = prever_esforco(inputs)
        
    grafico = gerar_grafico_regressao_importancia()
    
    contexto = {
        'inputs': inputs,
        'resultado': resultado,
        'metricas': metricas,
        'grafico': grafico,
    }
    return render(request, 'ml_regressao.html', contexto)


def ml_classificacao(request):
    """
    Laboratório de Classificação — Detecção Preventiva de Bugs Críticos em Produção.
    Exibe métricas formais (Acurácia, Precisão, Recall, F1, ROC-AUC) e permite ajuste do limiar.
    """
    cache = carregar_modelos()
    metricas = cache['metricas']['classificacao']
    
    inputs = {
        'num_regras_negocio': 6,
        'num_cenarios_bdd': 4,
        'num_casos_teste': 10,
        'num_edge_cases': 2,
        'num_integracoes': 2,
        'envolve_seguranca': 1,
        'tamanho_descricao': 180,
    }
    limiar = 0.35
    
    if request.method == 'POST':
        for campo in inputs.keys():
            val = request.POST.get(campo)
            if val is not None:
                inputs[campo] = int(val)
        try:
            limiar = float(request.POST.get('limiar', 0.35))
        except ValueError:
            limiar = 0.35
            
    resultado = prever_risco(inputs, limiar=limiar)
    grafico = gerar_grafico_classificacao_roc()
    
    contexto = {
        'inputs': inputs,
        'limiar': limiar,
        'resultado': resultado,
        'metricas': metricas,
        'grafico': grafico,
    }
    return render(request, 'ml_classificacao.html', contexto)


def ml_clusterizacao(request):
    """
    Laboratório de Clusterização — Agrupamento Não Supervisionado e Perfil de Requisitos.
    Apresenta Método do Cotovelo, Coeficiente de Silhueta, Redução Dimensional PCA 2D e Centróides.
    """
    cache = carregar_modelos()
    metricas = cache['metricas']['clusterizacao']
    requisitos_db = Requisito.objects.filter(processado=True)[:10]
    
    inputs = {
        'num_regras_negocio': 5,
        'num_cenarios_bdd': 3,
        'num_casos_teste': 8,
        'num_edge_cases': 3,
        'num_integracoes': 1,
        'envolve_seguranca': 0,
        'tamanho_descricao': 120,
    }
    req_selecionado_id = None
    
    if request.method == 'POST':
        req_id = request.POST.get('requisito_id')
        if req_id and req_id.isdigit():
            req_selecionado = Requisito.objects.filter(pk=int(req_id)).first()
            if req_selecionado:
                inputs = extrair_features_requisito(req_selecionado)
                req_selecionado_id = req_selecionado.pk
        else:
            for campo in inputs.keys():
                val = request.POST.get(campo)
                if val is not None:
                    inputs[campo] = int(val)
                    
    resultado = atribuir_perfil(inputs)
    grafico = gerar_grafico_clusterizacao_pca(destacar_ponto=resultado['coord_pca_2d'])
    
    contexto = {
        'inputs': inputs,
        'resultado': resultado,
        'metricas': metricas,
        'grafico': grafico,
        'requisitos_db': requisitos_db,
        'req_selecionado_id': req_selecionado_id,
    }
    return render(request, 'ml_clusterizacao.html', contexto)
