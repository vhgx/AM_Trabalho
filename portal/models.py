from django.db import models


class Requisito(models.Model):
    """
    Modelo que persiste cada requisito submetido e a resposta gerada pela IA.
    Rastreamento: Requisito -> Resposta Gerada (conforme proposta, seção 4).
    """
    titulo = models.CharField(max_length=200, verbose_name='Título')
    descricao = models.TextField(verbose_name='Requisito Funcional')
    interpretacao = models.TextField(blank=True, verbose_name='Interpretação do Requisito')
    bdd = models.TextField(blank=True, verbose_name='Cenários BDD (Gherkin)')
    casos_teste = models.TextField(blank=True, verbose_name='Casos de Teste Estruturados')
    playwright = models.TextField(blank=True, verbose_name='Código Playwright')
    edge_cases = models.TextField(blank=True, verbose_name='Edge Cases')
    criado_em = models.DateTimeField(auto_now_add=True, verbose_name='Criado em')
    atualizado_em = models.DateTimeField(auto_now=True, verbose_name='Atualizado em')
    processado = models.BooleanField(default=False, verbose_name='Processado')

    def __str__(self):
        return self.titulo

    class Meta:
        ordering = ['-criado_em']
        verbose_name = 'Requisito'
        verbose_name_plural = 'Requisitos'
