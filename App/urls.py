from django.urls import path
from . import views

urlpatterns = [
    path('', views.main, name='main'),
    path('fit/', views.fit, name='fit'),
    path('other_fit', views.other_fit, name='other fit'),
    path('extract/', views.extract, name='extract'),
    path('about/', views.about, name='about'),
]