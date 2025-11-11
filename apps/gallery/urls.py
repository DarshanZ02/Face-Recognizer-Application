from django.urls import path
from . import views

urlpatterns = [
    path('', views.gallery_home, name='gallery_home'),
]
