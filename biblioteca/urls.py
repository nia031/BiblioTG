from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('rec_contenido/<title>/', views.rec_contenido, name='rec_contenido'),
]

