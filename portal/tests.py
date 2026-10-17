from django.test import TestCase, Client
from django.urls import reverse

from .models import Requisito
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


class MLFeaturesTests(TestCase):
    def test_extracao_features_requisito(self):
        class MockRequisito:
            descricao = "O cliente precisa de autenticação via token JWT para consultar a API de pagamentos PIX."
            interpretacao = "### Regras de Negócio\n- Validar token\n- Expirar em 10 minutos\n- Limite de 3 tentativas"
            bdd = "Feature: Autenticação\nScenario: Login com sucesso"
            casos_teste = "| ID | Ação | Resultado |\n|---|---|---|\n| CT01 | Token válido | 200 OK |\n| CT02 | Token expirado | 401 |"
            edge_cases = "- Queda no gateway"

        feats = extrair_features_requisito(MockRequisito())
        self.assertEqual(feats['num_regras_negocio'], 3)
        self.assertEqual(feats['num_casos_teste'], 2)
        self.assertGreaterEqual(feats['num_integracoes'], 2)
        self.assertEqual(feats['envolve_seguranca'], 1)


class MLServicesTests(TestCase):
    def setUp(self):
        self.features_exemplo = {
            'num_regras_negocio': 5,
            'num_casos_teste': 8,
            'num_integracoes': 2,
            'envolve_seguranca': 0,
        }

    def test_prever_esforco(self):
        res = prever_esforco(self.features_exemplo)
        self.assertIn('esforco_horas', res)
        self.assertGreater(res['esforco_horas'], 0)
        self.assertGreater(res['metricas']['r2'], 0.80)

    def test_prever_risco(self):
        res = prever_risco(self.features_exemplo)
        self.assertIn('probabilidade_bug', res)
        self.assertIn(res['nivel_risco'], ['Baixo', 'Médio', 'Alto'])

    def test_atribuir_perfil(self):
        res = atribuir_perfil(self.features_exemplo)
        self.assertIn(res['cluster_id'], [0, 1, 2, 3])
        self.assertIn(res['nome'], ['CRUD Simples', 'Fluxo de Negócio', 'Integração Externa', 'Crítico / Segurança'])

    def test_obter_insights_requisito(self):
        req = Requisito.objects.create(
            titulo="Consulta de Saldo via API",
            descricao="Endpoint para consulta de saldo com token JWT.",
            interpretacao="### Regras de Negócio\n- Saldo em reais\n- Apenas clientes ativos",
            bdd="Scenario: Consulta com sucesso",
            casos_teste="| ID | Descrição |\n|---|---|\n| 1 | Consulta válida |",
            edge_cases="- Timeout de conexão",
            processado=True,
        )
        insights = obter_insights_requisito(req)
        self.assertIn('regressao', insights)
        self.assertIn('classificacao', insights)
        self.assertIn('clusterizacao', insights)
        self.assertIn('features', insights)

    def test_graficos_base64(self):
        g_reg = gerar_grafico_regressao_importancia()
        g_clust = gerar_grafico_clusterizacao(self.features_exemplo)
        self.assertTrue(len(g_reg) > 100)
        self.assertTrue(len(g_clust) > 100)


class MLViewsTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_view_resultado_exibe_estimativas(self):
        req = Requisito.objects.create(
            titulo="Pagamento PIX",
            descricao="Integração com API PIX.",
            interpretacao="### Regras de Negócio\n- Chave válida",
            bdd="Scenario: PIX OK",
            casos_teste="| ID | Ação |\n|---|---|\n| 1 | Transfere |",
            edge_cases="- Chave inexistente",
            processado=True,
        )
        resp = self.client.get(reverse('resultado', kwargs={'pk': req.pk}))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Estimativas (ML)')
        self.assertContains(resp, 'Esforço Estimado')

    def test_views_ml_get_e_post(self):
        payload = {
            'num_regras_negocio': 6,
            'num_casos_teste': 10,
            'num_integracoes': 2,
            'envolve_seguranca': 1,
        }
        for route_name in ['ml_regressao', 'ml_classificacao', 'ml_clusterizacao']:
            url = reverse(route_name)
            self.assertEqual(self.client.get(url).status_code, 200)
            self.assertEqual(self.client.post(url, payload).status_code, 200)
