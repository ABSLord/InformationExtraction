from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .Core.core import fit, predict
import json

LANG = 'rus'

#  главная страница
@csrf_exempt
def main(request):
    return render(
        request,
        'main.html',
        {})

# обучает новую модель
@csrf_exempt
def fit(request):
    if request.method == "POST":
        fit(request.POST['model_name'], request.FILES, json.loads(request.POST['coords']))
        return HttpResponse("Модель обучена")

    return HttpResponse("Метод не поддерживается")


# выделяет что-то с помощью уже обученной модели
@csrf_exempt
def extract(request):
    return render(
        request,
        'extract.html',
    )


# страница со справкой
@csrf_exempt
def about(request):
    return render(
        request,
        'about.html',
    )
