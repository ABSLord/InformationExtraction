from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.http import JsonResponse
from django.core import serializers
from .Core.core import train_and_save_model, predict
from .models import ExtractionModel
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
        train_and_save_model(request.POST['model_name'], request.FILES, json.loads(request.POST['coords']))
        return HttpResponse("Модель обучена")

    return HttpResponse("Метод не поддерживается")


# выделяет что-то с помощью уже обученной модели
@csrf_exempt
def extract(request):
    if request.method == "POST":
        model = request.POST['model'].strip()
        format = request.POST['format'].strip()
        img = list(request.FILES.items())[0]
        words = predict(model,img)
        return render(
            request,
            'text.html',
            {'words': ' '.join(words)}
        )
    return render(
        request,
        'extract.html',
    )


@csrf_exempt
def get_models(request):
    if request.method == "GET":
        values = ExtractionModel.objects.values('name')
        return JsonResponse(list(values), safe=False)
    return HttpResponse("Метод не поддерживается")


# страница со справкой
@csrf_exempt
def about(request):
    return render(
        request,
        'about.html',
    )
