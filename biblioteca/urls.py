from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    #path('rec_contenido/', views.rec_contenido, name='rec_contenido'),
    path('rec_contenido/<title>/', views.rec_contenido, name='rec_contenido'),
]

    #url(r'^drinks/(?P<drink_name>\D+)/',TemplateView.as_view(template_name='drinks/index.html')),
