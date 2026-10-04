from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import Requisito
from .services import run_qa_pipeline
from .ml_services import (
    prever_esforco,
    gerar_grafico_importancia,
    classificar_prioridade,
    executar_clusterizacao,
    extrair_features_requisito,
    gerar_dataset_clusterizacao_sintetico,
    FEATURES_REGRESSAO,
    FEATURES_CLASSIFICACAO,
    CLASSES_PRIORIDADE,
)


def dashboard(request):
    """Painel Central (Dashboard) - exibe requisitos submetidos anteriormente."""
    requisitos = Requisito.objects.all()
    dicionario = {'requisitos': requisitos}
    return render(request, 'dashboard.html', dicionario)


def submeter(request):
    """Submissão de Contexto - formulário para especificar o requisito."""
    if request.method == 'POST':
        titulo = request.POST.get('titulo', '').strip()
        descricao = request.POST.get('descricao', '').strip()

        if titulo and descricao:
            requisito = Requisito(titulo=titulo, descricao=descricao)
            requisito.save()
            return redirect('processar', pk=requisito.pk)

    return render(request, 'submeter.html')


def processar(request, pk):
    """Dispara o pipeline de IA para processar o requisito."""
    requisito = get_object_or_404(Requisito, pk=pk)

    if not requisito.processado:
        try:
            resultados = run_qa_pipeline(requisito.descricao)
            requisito.interpretacao = resultados['interpretacao']
            requisito.bdd = resultados['bdd']
            requisito.casos_teste = resultados['casos_teste']
            requisito.playwright = resultados['playwright']
            requisito.edge_cases = resultados['edge_cases']
            requisito.processado = True
            requisito.save()
        except Exception as e:
            dicionario = {'requisito': requisito, 'erro': str(e)}
            return render(request, 'erro.html', dicionario)

    return redirect('resultado', pk=requisito.pk)


def resultado(request, pk):
    """Exibição Rica e Multidimensional dos resultados gerados."""
    requisito = get_object_or_404(Requisito, pk=pk)
    dicionario = {'requisito': requisito}
    return render(request, 'resultado.html', dicionario)


# =====================================================================
# Views de Machine Learning
# =====================================================================


def ml_regressao(request):
    """
    Página de Regressão — Prever Esforço de Teste (horas).
    GET: exibe formulário. POST: treina modelo, faz previsão, retorna resultado.
    """
    contexto = {}

    if request.method == 'POST':
        try:
            complexidade = int(request.POST.get('complexidade', 5))
            num_integracoes = int(request.POST.get('num_integracoes', 2))
            tamanho_requisito = int(request.POST.get('tamanho_requisito', 200))
            num_regras_negocio = int(request.POST.get('num_regras_negocio', 3))

            resultado = prever_esforco(
                complexidade, num_integracoes, tamanho_requisito, num_regras_negocio
            )
            grafico_base64 = gerar_grafico_importancia()

            contexto = {
                'resultado': resultado,
                'grafico': grafico_base64,
                'inputs': {
                    'complexidade': complexidade,
                    'num_integracoes': num_integracoes,
                    'tamanho_requisito': tamanho_requisito,
                    'num_regras_negocio': num_regras_negocio,
                },
            }
        except Exception as e:
            contexto['erro'] = str(e)

    return render(request, 'ml_regressao.html', contexto)


def ml_classificacao(request):
    """
    Página de Classificação — Classificar Prioridade de Bug/Requisito.
    GET: exibe formulário. POST: classifica e retorna resultado.
    """
    contexto = {}

    if request.method == 'POST':
        try:
            impacto_usuario = int(request.POST.get('impacto_usuario', 5))
            complexidade_tecnica = int(request.POST.get('complexidade_tecnica', 5))
            frequencia_uso = int(request.POST.get('frequencia_uso', 5))
            tem_dependencia_externa = int(request.POST.get('tem_dependencia_externa', 0))
            risco_seguranca = int(request.POST.get('risco_seguranca', 5))

            resultado = classificar_prioridade(
                impacto_usuario, complexidade_tecnica, frequencia_uso,
                tem_dependencia_externa, risco_seguranca
            )

            contexto = {
                'resultado': resultado,
                'inputs': {
                    'impacto_usuario': impacto_usuario,
                    'complexidade_tecnica': complexidade_tecnica,
                    'frequencia_uso': frequencia_uso,
                    'tem_dependencia_externa': tem_dependencia_externa,
                    'risco_seguranca': risco_seguranca,
                },
            }
        except Exception as e:
            contexto['erro'] = str(e)

    return render(request, 'ml_classificacao.html', contexto)


def ml_clusterizacao(request):
    """
    Página de Clusterização — Agrupar Requisitos Similares.
    Usa requisitos do banco se houver ≥5, senão usa dados sintéticos.
    """
    n_clusters = int(request.POST.get('n_clusters', 3)) if request.method == 'POST' else 3
    contexto = {'n_clusters': n_clusters}

    try:
        # Tentar usar dados reais do banco
        requisitos = Requisito.objects.all()
        if requisitos.count() >= 5:
            dados_features = [extrair_features_requisito(r) for r in requisitos]
            titulos = [r.titulo for r in requisitos]
            fonte_dados = 'banco'
        else:
            dados_features, titulos = gerar_dataset_clusterizacao_sintetico()
            fonte_dados = 'sintetico'

        resultado = executar_clusterizacao(dados_features, titulos, n_clusters)

        # Definir cores para os clusters
        cores_cluster = ['primary', 'success', 'warning', 'danger', 'info', 'secondary']
        for item in resultado['resultados']:
            item['cor'] = cores_cluster[item['cluster'] % len(cores_cluster)]

        contexto['resultado'] = resultado
        contexto['fonte_dados'] = fonte_dados
    except Exception as e:
        contexto['erro'] = str(e)

    return render(request, 'ml_clusterizacao.html', contexto)

