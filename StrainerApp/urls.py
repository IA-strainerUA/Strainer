from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('procesar_formulario', views.procesar_formulario, name='procesar_formulario'),
    path('resultados', views.resultados, name='resultados'),
]
