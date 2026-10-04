from django.test import TestCase, Client
from django.urls import reverse

from .ml_services import (
    prever_esforco,
    classificar_prioridade,
    executar_clusterizacao,
    gerar_dataset_clusterizacao_sintetico,
    gerar_grafico_importancia,
    CLASSES_PRIORIDADE,
)


class MLRegressaoViewTests(TestCase):
    """Testes para a view de Regressão."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('ml_regressao')

    def test_get_regressao_retorna_200(self):
        """GET na página de regressão deve retornar status 200."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Regressão')

    def test_post_regressao_retorna_previsao(self):
        """POST com dados válidos deve retornar uma previsão numérica."""
        response = self.client.post(self.url, {
            'complexidade': 7,
            'num_integracoes': 3,
            'tamanho_requisito': 500,
            'num_regras_negocio': 5,
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('resultado', response.context)
        resultado = response.context['resultado']
        self.assertIn('previsao', resultado)
        self.assertIsInstance(resultado['previsao'], float)
        self.assertGreater(resultado['previsao'], 0)

    def test_post_regressao_retorna_metricas(self):
        """POST deve retornar métricas do modelo (R², MAE, RMSE)."""
        response = self.client.post(self.url, {
            'complexidade': 5,
            'num_integracoes': 2,
            'tamanho_requisito': 200,
            'num_regras_negocio': 3,
        })
        resultado = response.context['resultado']
        self.assertIn('metricas', resultado)
        self.assertIn('r2', resultado['metricas'])
        self.assertIn('mae', resultado['metricas'])
        self.assertIn('rmse', resultado['metricas'])

    def test_post_regressao_retorna_grafico(self):
        """POST deve retornar um gráfico base64 para exibição."""
        response = self.client.post(self.url, {
            'complexidade': 5,
            'num_integracoes': 2,
            'tamanho_requisito': 200,
            'num_regras_negocio': 3,
        })
        self.assertIn('grafico', response.context)
        self.assertTrue(len(response.context['grafico']) > 100)


class MLClassificacaoViewTests(TestCase):
    """Testes para a view de Classificação."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('ml_classificacao')

    def test_get_classificacao_retorna_200(self):
        """GET na página de classificação deve retornar status 200."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Classificação')

    def test_post_classificacao_retorna_prioridade_valida(self):
        """POST com dados válidos deve retornar uma prioridade válida."""
        response = self.client.post(self.url, {
            'impacto_usuario': 9,
            'complexidade_tecnica': 6,
            'frequencia_uso': 8,
            'tem_dependencia_externa': 1,
            'risco_seguranca': 7,
        })
        self.assertEqual(response.status_code, 200)
        resultado = response.context['resultado']
        self.assertIn(resultado['prioridade'], CLASSES_PRIORIDADE)

    def test_post_classificacao_retorna_probabilidades(self):
        """POST deve retornar probabilidades que somam ~100%."""
        response = self.client.post(self.url, {
            'impacto_usuario': 5,
            'complexidade_tecnica': 5,
            'frequencia_uso': 5,
            'tem_dependencia_externa': 0,
            'risco_seguranca': 5,
        })
        resultado = response.context['resultado']
        total = sum(resultado['probabilidades'].values())
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_post_classificacao_retorna_metricas(self):
        """POST deve retornar métricas do modelo (acurácia e F1)."""
        response = self.client.post(self.url, {
            'impacto_usuario': 5,
            'complexidade_tecnica': 5,
            'frequencia_uso': 5,
            'tem_dependencia_externa': 0,
            'risco_seguranca': 5,
        })
        resultado = response.context['resultado']
        self.assertIn('acuracia', resultado['metricas'])
        self.assertIn('f1_macro', resultado['metricas'])


class MLClusterizacaoViewTests(TestCase):
    """Testes para a view de Clusterização."""

    def setUp(self):
        self.client = Client()
        self.url = reverse('ml_clusterizacao')

    def test_get_clusterizacao_retorna_200(self):
        """GET na página de clusterização deve retornar status 200."""
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Clusterização')

    def test_clusterizacao_com_dados_sinteticos(self):
        """Sem requisitos no banco, deve usar dados sintéticos."""
        response = self.client.get(self.url)
        self.assertEqual(response.context['fonte_dados'], 'sintetico')

    def test_post_clusterizacao_com_n_clusters(self):
        """POST com n_clusters deve retornar o número correto de clusters."""
        response = self.client.post(self.url, {'n_clusters': 4})
        self.assertEqual(response.status_code, 200)
        resultado = response.context['resultado']
        self.assertEqual(resultado['n_clusters'], 4)

    def test_clusterizacao_retorna_centroides(self):
        """A clusterização deve retornar centróides para cada cluster."""
        response = self.client.post(self.url, {'n_clusters': 3})
        resultado = response.context['resultado']
        self.assertEqual(len(resultado['centroides']), 3)


class MLServicesUnitTests(TestCase):
    """Testes unitários para as funções do ml_services.py"""

    def test_prever_esforco_retorna_dict_com_previsao(self):
        """prever_esforco deve retornar dict com chave 'previsao'."""
        resultado = prever_esforco(5, 3, 300, 4)
        self.assertIn('previsao', resultado)
        self.assertIsInstance(resultado['previsao'], float)

    def test_classificar_prioridade_retorna_classe_valida(self):
        """classificar_prioridade deve retornar uma classe válida."""
        resultado = classificar_prioridade(8, 7, 9, 1, 8)
        self.assertIn(resultado['prioridade'], CLASSES_PRIORIDADE)

    def test_executar_clusterizacao_agrupa_corretamente(self):
        """executar_clusterizacao deve atribuir clusters a todos os itens."""
        dados, titulos = gerar_dataset_clusterizacao_sintetico(n_amostras=20)
        resultado = executar_clusterizacao(dados, titulos, n_clusters=3)
        self.assertEqual(len(resultado['resultados']), 20)
        self.assertEqual(resultado['n_clusters'], 3)

    def test_gerar_grafico_importancia_retorna_base64(self):
        """gerar_grafico_importancia deve retornar uma string base64 válida."""
        img = gerar_grafico_importancia()
        self.assertIsInstance(img, str)
        self.assertTrue(len(img) > 100)
