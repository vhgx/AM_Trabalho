from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('submeter/', views.submeter, name='submeter'),
    path('processar/<int:pk>/', views.processar, name='processar'),
    path('resultado/<int:pk>/', views.resultado, name='resultado'),
    # Machine Learning
    path('ml/regressao/', views.ml_regressao, name='ml_regressao'),
    path('ml/classificacao/', views.ml_classificacao, name='ml_classificacao'),
    path('ml/clusterizacao/', views.ml_clusterizacao, name='ml_clusterizacao'),
]

