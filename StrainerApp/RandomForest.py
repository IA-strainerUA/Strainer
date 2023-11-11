import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Cargar datos desde el archivo Excel
data = pd.read_excel("datos_limpios_sin_preguntas_2.xlsx")

# Suponiendo que la última columna es la variable de clase y las demás son atributos
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Contar las muestras por clase
samples_per_class = y.value_counts().to_dict()

# Obtener la cantidad de muestras de la clase minoritaria
min_samples = min(samples_per_class.values())

# Definir el número de vecinos para SMOTE
n_neighbors = min(6, min_samples - 1)  # Ajustar el número de vecinos para que sea igual o menor a la cantidad de muestras

# Aplicar SMOTE con el número ajustado de vecinos
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Inicializar el clasificador de Random Forest
clf = RandomForestClassifier()

# Entrenar el modelo
clf.fit(X_train, y_train)

# Predicción en el conjunto de prueba
y_pred = clf.predict(X_test)

# Matriz de Confusión
conf_matrix = confusion_matrix(y_test, y_pred)

# Análisis de métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Gráfico de importancia de atributos
feature_importances = clf.feature_importances_
indices = feature_importances.argsort()[::-1]

plt.figure()
plt.title("Importancia de Atributos")
plt.bar(range(X.shape[1]), feature_importances[indices])
plt.xticks(range(X.shape[1]), indices)
plt.show()

# Curva ROC
y_pred_prob = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# Mostrar resultados
print("Matriz de Confusión:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Pide al usuario ingresar los valores de los atributos por la terminal
nuevo_dato = {}
for column in X.columns:  # Incluye todas las columnas
    valor = input(f"Ingrese el valor para '{column}': ")
    nuevo_dato[column] = float(valor) if valor.isnumeric() else valor

# Crea un DataFrame con el nuevo dato
nuevo_dato_df = pd.DataFrame([nuevo_dato])

# Utiliza el modelo de Random Forest para predecir
prediccion = clf.predict(nuevo_dato_df)
probabilidades = clf.predict_proba(nuevo_dato_df)

# La probabilidad de ser de la clase 0 es 1 menos la probabilidad de ser de la clase 1
probabilidad_clase_0 = 1 - probabilidades[0, 1]

# Convierte la probabilidad a un porcentaje
probabilidad_porcentaje = probabilidad_clase_0 * 100

# Imprime la probabilidad de ser clase 0 como un porcentaje
print(f"La probabilidad de que el estudiante deserte de la carrera es de: {probabilidad_porcentaje:.2f}%")
