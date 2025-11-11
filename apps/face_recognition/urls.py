from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_images, name='upload_images'),
    path('download-zip/', views.download_zip, name='download_zip'),
]

