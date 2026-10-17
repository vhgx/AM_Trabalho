from django.shortcuts import render, redirect, get_object_or_404
from .models import Requisito
from .services import run_qa_pipeline
from .ml.features import extrair_features_requisito
from .ml_services import (
    carregar_modelos,
    prever_esforco,
    prever_risco,
    atribuir_perfil,
    obter_insights_requisito,
    gerar_grafico_regressao_importancia,
    gerar_grafico_clusterizacao,
)


def dashboard(request):
    """Painel Central (Dashboard) - exibe requisitos submetidos anteriormente."""
    requisitos = Requisito.objects.all()
    return render(request, 'dashboard.html', {'requisitos': requisitos})


def submeter(request):
    """Submissão de Contexto - recebe a história de usuário ou regra de negócio."""
    if request.method == 'POST':
        titulo = request.POST.get('titulo', '').strip()
        descricao = request.POST.get('descricao', '').strip()

        if not titulo or not descricao:
            return render(request, 'submeter.html', {
                'erro': 'Título e Requisito Funcional são campos obrigatórios.',
                'titulo': titulo,
                'descricao': descricao,
            })

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
            return render(request, 'erro.html', {
                'erro': f'Falha na comunicação com o provedor de IA: {str(e)}',
                'requisito': requisito,
            })

    return redirect('resultado', pk=requisito.pk)


def resultado(request, pk):
    """Exibe os artefatos de teste gerados e o resumo de estimativas via ML."""
    requisito = get_object_or_404(Requisito, pk=pk)
    insights = None

    if requisito.processado:
        try:
            insights = obter_insights_requisito(requisito)
        except Exception as e:
            insights = {'erro': f'Não foi possível calcular as estimativas: {str(e)}'}

    return render(request, 'resultado.html', {
        'requisito': requisito,
        'insights': insights,
    })


def _ler_inputs_formulario(request, padrao):
    """Lê os 4 parâmetros do formulário ou carrega de um requisito salvo."""
    inputs = dict(padrao)
    req_selecionado_id = None

    if request.method == 'POST':
        req_id = request.POST.get('requisito_id')
        if req_id and req_id.isdigit():
            req = Requisito.objects.filter(pk=int(req_id)).first()
            if req:
                inputs = extrair_features_requisito(req)
                req_selecionado_id = req.pk
        else:
            for campo in inputs.keys():
                val = request.POST.get(campo)
                if val is not None:
                    try:
                        inputs[campo] = int(val)
                    except ValueError:
                        pass

    return inputs, req_selecionado_id


def ml_regressao(request):
    """Previsão de esforço de teste em horas (Regressão)."""
    cache = carregar_modelos()
    padrao = {
        'num_regras_negocio': 4,
        'num_casos_teste': 8,
        'num_integracoes': 1,
        'envolve_seguranca': 0,
    }
    inputs, req_selecionado_id = _ler_inputs_formulario(request, padrao)
    resultado_reg = prever_esforco(inputs)
    grafico = gerar_grafico_regressao_importancia()

    return render(request, 'ml_regressao.html', {
        'inputs': inputs,
        'resultado': resultado_reg,
        'metricas': cache['metricas']['regressao'],
        'grafico': grafico,
        'requisitos_db': Requisito.objects.filter(processado=True)[:10],
        'req_selecionado_id': req_selecionado_id,
    })


def ml_classificacao(request):
    """Classificação de risco de defeito em produção (Supervisionado)."""
    cache = carregar_modelos()
    padrao = {
        'num_regras_negocio': 6,
        'num_casos_teste': 10,
        'num_integracoes': 2,
        'envolve_seguranca': 1,
    }
    inputs, req_selecionado_id = _ler_inputs_formulario(request, padrao)
    resultado_clf = prever_risco(inputs)

    return render(request, 'ml_classificacao.html', {
        'inputs': inputs,
        'resultado': resultado_clf,
        'metricas': cache['metricas']['classificacao'],
        'requisitos_db': Requisito.objects.filter(processado=True)[:10],
        'req_selecionado_id': req_selecionado_id,
    })


def ml_clusterizacao(request):
    """Agrupamento de requisitos por perfil (Não Supervisionado - K-Means)."""
    cache = carregar_modelos()
    padrao = {
        'num_regras_negocio': 5,
        'num_casos_teste': 8,
        'num_integracoes': 1,
        'envolve_seguranca': 0,
    }
    inputs, req_selecionado_id = _ler_inputs_formulario(request, padrao)
    resultado_clust = atribuir_perfil(inputs)
    grafico = gerar_grafico_clusterizacao(features_atuais=inputs)

    return render(request, 'ml_clusterizacao.html', {
        'inputs': inputs,
        'resultado': resultado_clust,
        'metricas': cache['metricas']['clusterizacao'],
        'grafico': grafico,
        'requisitos_db': Requisito.objects.filter(processado=True)[:10],
        'req_selecionado_id': req_selecionado_id,
    })
