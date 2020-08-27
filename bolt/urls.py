from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from web_project import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.contrib.auth import views as Aviews

urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('delete/<str:dir>/<str:pdf>', views.delete,name='delete'),
    path('compare/<str:dir>', views.compare, name='compare'),
    path('download/<str:result>', views.download, name='download'),
    path('save', views.save, name='save'),
]
urlpatterns += staticfiles_urlpatterns()