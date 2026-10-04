"""
Comando de Gerenciamento Django: treinar_modelos
Executa o treinamento formal dos modelos de Regressão, Classificação e Clusterização
e salva os artefatos serializados (.joblib) e métricas (.json) em 'ml_models/'.
"""
from django.core.management.base import BaseCommand
from portal.ml_services import treinar_e_salvar_modelos

class Command(BaseCommand):
    help = 'Treina os modelos de Machine Learning (Regressão, Classificação, Clusterização) a partir de historico_requisitos.csv'

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE('Iniciando treinamento dos modelos de Machine Learning...'))
        metricas = treinar_e_salvar_modelos()
        
        self.stdout.write(self.style.SUCCESS('\n[1/3] REGRESSÃO (Previsão de Esforço de Teste):'))
        self.stdout.write(f"  - Linear Regression: R² = {metricas['regressao']['linear']['r2']}, MAE = {metricas['regressao']['linear']['mae']}h, RMSE = {metricas['regressao']['linear']['rmse']}h")
        self.stdout.write(f"  - Random Forest:     R² = {metricas['regressao']['rf']['r2']}, MAE = {metricas['regressao']['rf']['mae']}h, RMSE = {metricas['regressao']['rf']['rmse']}h")
        
        self.stdout.write(self.style.SUCCESS('\n[2/3] CLASSIFICAÇÃO (Detecção de Risco de Bugs em Produção):'))
        self.stdout.write(f"  - Regressão Logística: Acurácia = {metricas['classificacao']['logistica']['accuracy']}, Recall = {metricas['classificacao']['logistica']['recall']}, F1 = {metricas['classificacao']['logistica']['f1']}, ROC-AUC = {metricas['classificacao']['logistica']['roc_auc']}")
        self.stdout.write(f"  - Árvore de Decisão:   Acurácia = {metricas['classificacao']['arvore']['accuracy']}, Recall = {metricas['classificacao']['arvore']['recall']}, F1 = {metricas['classificacao']['arvore']['f1']}, ROC-AUC = {metricas['classificacao']['arvore']['roc_auc']}")
        
        self.stdout.write(self.style.SUCCESS('\n[3/3] CLUSTERIZAÇÃO (Perfis de Requisitos):'))
        self.stdout.write(f"  - K ótimo = {metricas['clusterizacao']['k_otimo']}")
        self.stdout.write(f"  - Inércia (WCSS) = {metricas['clusterizacao']['inercia']}, Silhueta = {metricas['clusterizacao']['silhouette']}")
        self.stdout.write(f"  - Variância Explicada pelo PCA 2D: {metricas['clusterizacao']['variancia_total_retida']*100:.1f}%")
        
        self.stdout.write(self.style.SUCCESS('\nModelos e métricas gravados com sucesso no diretório ml_models/.'))
