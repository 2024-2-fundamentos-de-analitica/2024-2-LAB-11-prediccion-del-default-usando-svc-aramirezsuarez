# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#



import pandas as pd
import numpy as np
import os
import json
import gzip
import joblib
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def preprocess_data(zip_file_path):
    """ Limpieza de datos según las especificaciones. """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        csv_filename = zip_ref.namelist()[0]
        with zip_ref.open(csv_filename) as f:
            df = pd.read_csv(f)

    # Renombrar la variable objetivo y eliminar la columna "ID"
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)

    # Eliminar valores faltantes
    df.dropna(inplace=True)

    # Agrupar EDUCATION > 4 en la categoría "others"
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def split_data(df):
    """ Divide los datos en X (variables explicativas) e y (objetivo) """
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


def build_pipeline():
    """ Crea un pipeline con OneHotEncoding, PCA, StandardScaler, SelectKBest y SVM """

    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]

    preprocessor = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("scaler", StandardScaler(), slice(0, -1))  # Escalar todas las columnas excepto la última
    ], remainder="passthrough")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("pca", PCA()),  # Mantiene todas las componentes
        ("select", SelectKBest(f_classif, k=10)),  # Selecciona las 10 características más importantes
        ("classifier", SVC(kernel="rbf", probability=True))
    ])

    return pipeline


def optimize_hyperparameters(pipeline, X_train, y_train):
    """ Optimiza hiperparámetros usando GridSearchCV con validación cruzada """
    param_grid = {
        "pca__n_components": [10, 15, 20],  # Ajustar número de componentes principales
        "classifier__C": [0.1, 1, 10],  # Ajuste del hiperparámetro C de SVM
        "classifier__gamma": ["scale", "auto"]  # Ajuste del parámetro gamma de SVM
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"✅ Mejor precisión encontrada: {grid_search.best_score_}")
    print(f"🔍 Mejores hiperparámetros: {grid_search.best_params_}")

    return grid_search  # Retorna el GridSearchCV completo


def save_model(model, file_path):
    """ Guarda el modelo optimizado en gzip """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with gzip.open(file_path, "wb") as f:
        joblib.dump(model, f)


def calculate_metrics(model, X, y, dataset_type):
    """ Calcula precisión, recall, f1-score y matriz de confusión """
    y_pred = model.predict(X)

    metrics = {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }

    return metrics, cm_dict


def save_metrics(metrics_list, file_path):
    """ Guarda las métricas en un archivo JSON con el orden correcto """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    ordered_metrics = []  # Lista ordenada de métricas

    # Añadir primero las métricas
    for metric in metrics_list:
        if metric["type"] == "metrics":
            ordered_metrics.append(metric)

    # Luego añadir las matrices de confusión
    for metric in metrics_list:
        if metric["type"] == "cm_matrix":
            ordered_metrics.append(metric)

    # Guardar en JSON
    with open(file_path, "w", encoding="utf-8") as f:
        for metric in ordered_metrics:
            f.write(json.dumps(metric) + "\n")


def main():
    # Paso 1: Cargar y limpiar datos
    train_file = "../files/input/train_data.csv.zip"
    test_file = "../files/input/test_data.csv.zip"

    train_df = preprocess_data(train_file)
    test_df = preprocess_data(test_file)

    # Paso 2: Dividir datos en X e y
    X_train, y_train = split_data(train_df)
    X_test, y_test = split_data(test_df)

    # Paso 3: Construir pipeline
    pipeline = build_pipeline()

    # Paso 4: Optimizar hiperparámetros
    model = optimize_hyperparameters(pipeline, X_train, y_train)

    # Paso 5: Guardar modelo
    model_path = "../files/models/model.pkl.gz"
    save_model(model, model_path)

    # Paso 6 y 7: Calcular métricas y matriz de confusión
    metrics_train, cm_train = calculate_metrics(model.best_estimator_, X_train, y_train, "train")
    metrics_test, cm_test = calculate_metrics(model.best_estimator_, X_test, y_test, "test")

    # Guardar métricas en JSON
    metrics_path = "../files/output/metrics.json"
    save_metrics([metrics_train, cm_train, metrics_test, cm_test], metrics_path)

    print(f"✅ Modelo guardado en {model_path}. Métricas en {metrics_path}.")


if __name__ == "__main__":
    main()# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import pandas as pd
import numpy as np
import os
import json
import gzip
import joblib
import zipfile

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix


def limpiar_datos(ruta_zip):
    with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
        nombre_csv = zip_ref.namelist()[0]
        with zip_ref.open(nombre_csv) as f:
            df = pd.read_csv(f)
    
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df.dropna(inplace=True)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    
    return df


def dividir_datos(df):
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y


def construir_pipeline():
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocesador = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columnas_categoricas),
        ("scaler", StandardScaler(), slice(0, -1))
    ], remainder="passthrough")
    
    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("pca", PCA()),
        ("seleccion", SelectKBest(f_classif, k=10)),
        ("clasificador", SVC(kernel="rbf", probability=True))
    ])
    
    return pipeline


def optimizar_hiperparametros(pipeline, X_train, y_train):
    param_grid = {
        "pca__n_components": [10, 15, 20],
        "clasificador__C": [0.1, 1, 10],
        "clasificador__gamma": ["scale", "auto"]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring="balanced_accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


def guardar_modelo(modelo, ruta_archivo):
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    with gzip.open(ruta_archivo, "wb") as f:
        joblib.dump(modelo, f)


def calcular_metricas(modelo, X, y, tipo_datos):
    y_pred = modelo.predict(X)
    
    metricas = {
        "type": "metrics",
        "dataset": tipo_datos,
        "precision": precision_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }
    
    cm = confusion_matrix(y, y_pred)
    matriz_confusion = {
        "type": "cm_matrix",
        "dataset": tipo_datos,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])}
    }
    
    return metricas, matriz_confusion


def guardar_metricas(lista_metricas, ruta_archivo):
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    
    with open(ruta_archivo, "w", encoding="utf-8") as f:
        for metrica in lista_metricas:
            f.write(json.dumps(metrica) + "\n")


def main():
    ruta_train = "../files/input/train_data.csv.zip"
    ruta_test = "../files/input/test_data.csv.zip"
    
    df_train = limpiar_datos(ruta_train)
    df_test = limpiar_datos(ruta_test)
    
    X_train, y_train = dividir_datos(df_train)
    X_test, y_test = dividir_datos(df_test)
    
    pipeline = construir_pipeline()
    modelo = optimizar_hiperparametros(pipeline, X_train, y_train)
    
    ruta_modelo = "../files/models/model.pkl.gz"
    guardar_modelo(modelo, ruta_modelo)
    
    metricas_train, cm_train = calcular_metricas(modelo.best_estimator_, X_train, y_train, "train")
    metricas_test, cm_test = calcular_metricas(modelo.best_estimator_, X_test, y_test, "test")
    
    ruta_metricas = "../files/output/metrics.json"
    guardar_metricas([metricas_train, cm_train, metricas_test, cm_test], ruta_metricas)


if __name__ == "__main__":
    main()
