"""
Suíte de Testes Automatizados - QA Gen Insights
Cobre extração de features, serviços de predição dos 3 paradigmas de AM,
sanidade de métricas e views Django.
"""
from django.test import TestCase, Client
from django.urls import reverse

from .models import Requisito
from .ml.features import extrair_features_requisito, FEATURE_NAMES
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


class MLFeaturesTests(TestCase):
    """Testa a extração automatizada de atributos a partir de artefatos de QA."""

    def test_extracao_features_requisito(self):
        class MockRequisito:
            descricao = "O cliente precisa de autenticação via token JWT para consultar a API de pagamentos PIX."
            interpretacao = "### Regras de Negócio e Restrições\n- Deve validar assinatura SHA256\n- O token expira em 10 minutos\n- Limite de 3 tentativas"
            bdd = "Feature: Autenticação\nScenario: Login com sucesso\nGiven credenciais válidas\nThen 200 OK\n\nScenario Outline: Falhas\nGiven dados inválidos\nThen 401"
            casos_teste = "| ID | Ação | Resultado |\n|---|---|---|\n| CT01 | Informar token válido | 200 OK |\n| CT02 | Informar token expirado | 401 Unauthorized |"
            edge_cases = "- Queda no gateway durante o handshake\n- Payload corrompido com caracteres de injeção"

        req = MockRequisito()
        feats = extrair_features_requisito(req)

        self.assertEqual(feats['num_regras_negocio'], 3)
        self.assertEqual(feats['num_cenarios_bdd'], 2)
        self.assertEqual(feats['num_casos_teste'], 2)
        self.assertEqual(feats['num_edge_cases'], 2)
        self.assertGreaterEqual(feats['num_integracoes'], 2) # API, PIX, gateway
        self.assertEqual(feats['envolve_seguranca'], 1) # autenticação, token, jwt
        self.assertGreater(feats['tamanho_descricao'], 10)


class MLServicesUnitTests(TestCase):
    """Testa o comportamento numérico e saídas dos modelos de Machine Learning."""

    def setUp(self):
        self.features_exemplo = {
            'num_regras_negocio': 5,
            'num_cenarios_bdd': 3,
            'num_casos_teste': 8,
            'num_edge_cases': 3,
            'num_integracoes': 2,
            'envolve_seguranca': 0,
            'tamanho_descricao': 150,
        }

    def test_prever_esforco_retorna_horas_e_metricas(self):
        res = prever_esforco(self.features_exemplo)
        self.assertIn('esforco_horas', res)
        self.assertIsInstance(res['esforco_horas'], float)
        self.assertGreater(res['esforco_horas'], 0)
        self.assertIn('metricas', res)
        self.assertIn('rf', res['metricas'])
        self.assertGreater(res['metricas']['rf']['r2'], 0.85)

    def test_prever_risco_respeita_limiar_calibrado(self):
        # Testar com limiar estrito (0.20) e limiar conservador (0.80)
        res_alerta = prever_risco(self.features_exemplo, limiar=0.20)
        res_relaxado = prever_risco(self.features_exemplo, limiar=0.90)

        self.assertIn('probabilidade_bug', res_alerta)
        self.assertIn(res_alerta['nivel_risco'], ['Baixo', 'Médio', 'Alto'])
        self.assertTrue(res_alerta['gerou_alerta'])
        self.assertFalse(res_relaxado['gerou_alerta'])

    def test_atribuir_perfil_cluster_e_pca(self):
        res = atribuir_perfil(self.features_exemplo)
        self.assertIn(res['cluster_id'], [0, 1, 2, 3])
        self.assertIn(res['nome'], ['CRUD Simples', 'Fluxo de Negócio Complexo', 'Integração Pesada', 'Crítico / Segurança'])
        self.assertEqual(len(res['coord_pca_2d']), 2)
        self.assertIsInstance(res['coord_pca_2d'][0], float)

    def test_obter_insights_requisito_completo(self):
        req = Requisito.objects.create(
            titulo="Consulta de Saldo via API",
            descricao="Endpoint para consulta de saldo bancário com token JWT.",
            interpretacao="### Regras de Negócio\n- Saldo em reais\n- Apenas clientes ativos",
            bdd="Scenario: Consulta realizada com sucesso\nGiven token\nThen saldo",
            casos_teste="| ID | Descrição |\n|---|---|\n| 1 | Consulta válida |",
            edge_cases="- Timeout de conexão com o core bancário",
            processado=True
        )
        insights = obter_insights_requisito(req)
        self.assertIn('regressao', insights)
        self.assertIn('classificacao', insights)
        self.assertIn('clusterizacao', insights)
        self.assertIn('features', insights)
        self.assertIn('recomendacoes', insights)
        self.assertGreater(len(insights['recomendacoes']), 0)

    def test_graficos_geram_base64_valido(self):
        g_reg = gerar_grafico_regressao_importancia()
        g_clf = gerar_grafico_classificacao_roc()
        g_clust = gerar_grafico_clusterizacao_pca()

        self.assertIsInstance(g_reg, str)
        self.assertGreater(len(g_reg), 100)
        self.assertIsInstance(g_clf, str)
        self.assertGreater(len(g_clf), 100)
        self.assertIsInstance(g_clust, str)
        self.assertGreater(len(g_clust), 100)


class MLViewsIntegrationTests(TestCase):
    """Testa as rotas e renderização de templates com os novos componentes de ML."""

    def setUp(self):
        self.client = Client()

    def test_view_resultado_exibe_aba_insights(self):
        req = Requisito.objects.create(
            titulo="Pagamento PIX",
            descricao="Integração com API PIX do Banco Central.",
            interpretacao="### Regras de Negócio\n- Chave válida",
            bdd="Scenario: PIX OK\nGiven chave\nThen transferir",
            casos_teste="| ID | Ação |\n|---|---|\n| 1 | Transfere |",
            edge_cases="- Chave não cadastrada",
            processado=True
        )
        url = reverse('resultado', kwargs={'pk': req.pk})
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Insights de ML')
        self.assertContains(resp, 'Esforço Estimado')
        self.assertContains(resp, 'Risco de Defeito')
        self.assertContains(resp, 'Perfil do Requisito')

    def test_view_ml_regressao_get_e_post(self):
        url = reverse('ml_regressao')
        # GET
        resp_get = self.client.get(url)
        self.assertEqual(resp_get.status_code, 200)
        self.assertContains(resp_get, 'Laboratório de Regressão')

        # POST
        resp_post = self.client.post(url, {
            'num_regras_negocio': 7,
            'num_cenarios_bdd': 5,
            'num_casos_teste': 12,
            'num_edge_cases': 4,
            'num_integracoes': 3,
            'envolve_seguranca': 1,
            'tamanho_descricao': 220,
        })
        self.assertEqual(resp_post.status_code, 200)
        self.assertContains(resp_post, 'Estimativa de Esforço Previsto')

    def test_view_ml_classificacao_get_e_post(self):
        url = reverse('ml_classificacao')
        resp_get = self.client.get(url)
        self.assertEqual(resp_get.status_code, 200)
        self.assertContains(resp_get, 'Laboratório de Classificação')

        resp_post = self.client.post(url, {
            'num_regras_negocio': 8,
            'num_cenarios_bdd': 4,
            'num_casos_teste': 10,
            'num_edge_cases': 2,
            'num_integracoes': 4,
            'envolve_seguranca': 1,
            'tamanho_descricao': 250,
            'limiar': 0.30,
        })
        self.assertEqual(resp_post.status_code, 200)
        self.assertContains(resp_post, 'Diagnóstico de Risco')

    def test_view_ml_clusterizacao_get_e_post(self):
        url = reverse('ml_clusterizacao')
        resp_get = self.client.get(url)
        self.assertEqual(resp_get.status_code, 200)
        self.assertContains(resp_get, 'Laboratório de Clusterização')

        resp_post = self.client.post(url, {
            'num_regras_negocio': 3,
            'num_cenarios_bdd': 2,
            'num_casos_teste': 4,
            'num_edge_cases': 1,
            'num_integracoes': 0,
            'envolve_seguranca': 0,
            'tamanho_descricao': 80,
        })
        self.assertEqual(resp_post.status_code, 200)
        self.assertContains(resp_post, 'Cluster & Perfil de Requisito Identificado')


class MLMetricsSanityTests(TestCase):
    """Audita a consistência estatística mínima dos modelos treinados."""

    def test_metricas_minimas_garantidas(self):
        cache = carregar_modelos()
        meta = cache['metricas']

        # 1. Regressão: R² > 0.85
        self.assertGreater(meta['regressao']['rf']['r2'], 0.85)
        self.assertGreater(meta['regressao']['linear']['r2'], 0.85)

        # 2. Classificação: ROC-AUC > 0.70
        self.assertGreater(meta['classificacao']['logistica']['roc_auc'], 0.70)

        # 3. Clusterização: Silhueta > 0.35 e retenção PCA > 0.65
        self.assertGreater(meta['clusterizacao']['silhouette'], 0.35)
        self.assertGreater(meta['clusterizacao']['variancia_total_retida'], 0.65)
