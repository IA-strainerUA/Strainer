from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
import json
from django.http import JsonResponse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from django.template.response import TemplateResponse
import time

# Aquí debes definir la lógica para tu formulario y predicción
def index(request):
    return render(request, 'index.html', {'probabilidad': None}) 

def procesar_prediccion(datos_del_formulario):
    file_path = 'dataset/datos.xlsx'
    data = pd.read_excel(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)


    nuevo_dato_df = pd.DataFrame([datos_del_formulario])
    probabilidades = clf.predict_proba(nuevo_dato_df)

    probabilidad_clase_0 = 1 - probabilidades[0, 1]
    probabilidad_porcentaje = probabilidad_clase_0 * 100

    return probabilidad_porcentaje

# def prediccion(request):
#     if request.method == 'POST':
#         try:
#             body_unicode = request.body.decode('utf-8')
#             data = json.loads(body_unicode)
#             print(data)

#             probabilidad = procesar_prediccion(data)
#             print(type(probabilidad))
#             print(probabilidad)
#             time.sleep(1)
            
#             return render(request, 'resultados.html', {"probabilidad": 1})

#         except Exception as e:
#             print("Se produjo un error:", e)

#             # Aquí maneja el error como mejor convenga para tu aplicación
#             return render(request, 'resultados.html', {'probabilidad': None})

def procesar_formulario(request):
    if request.method == 'POST':
        data = request.POST.dict()
        data.pop('csrfmiddlewaretoken')

        # Puedes imprimir los datos para verificar
        probabilidad = procesar_prediccion(data)

        # Por ahora, devolvemos un valor de ejemplo

        return render(request, 'resultados.html', {"probabilidad": probabilidad})
    else:
        # Devuelve una respuesta vacía o maneja el caso de GET
        return JsonResponse({})

def resultados(request):
    return render(request, 'resultados.html', {"probabilidad": 1})