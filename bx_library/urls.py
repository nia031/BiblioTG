from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('buscar', views.buscar, name='buscar'),
    path('colaborativo/<query_book>/', views.colaborativo, name='colaborativo'),
]