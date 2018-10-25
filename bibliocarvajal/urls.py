
from django.contrib import admin
from django.urls import path, include
from biblioteca import views
from accounts import views
from bx_library import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('accounts/', include(('accounts.urls','accounts'), namespace='accounts')),
    path('biblioteca/', include(('biblioteca.urls','biblioteca'), namespace='biblioteca')),
    path('bx_library/', include(('bx_library.urls','bx_library'), namespace='bx_library')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
