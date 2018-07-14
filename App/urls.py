from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('fit/', views.fit, name='fit'),
    path('extract/', views.extract, name='extract'),
    path('get_models/', views.get_models, name='get_models'),
    path('about/', views.about, name='about'),
]