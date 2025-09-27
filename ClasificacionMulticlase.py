#%%
# Contenido del archivo:
#     Importaciones
#     Carga de datos
#     Funciones definidas
#     Separación del dataset en variables explicativas y variables a explicar
#     Exploración de datos
#     Clasificación binaria
#     Selección de Atributos
#     Clasificación Multiclase
#%% ===============================================================================================
# IMPORTACIONES 
# =================================================================================================
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, classification_report
import duckdb 
import random 
#%% ===============================================================================================
# CARGA DE DATOS 
# =================================================================================================
carpetaOriginal = os.path.dirname(os.path.abspath(__file__))
ruta_moda = os.path.join(carpetaOriginal, "Fashion-MNIST.csv", ) 
fashion = pd.read_csv(ruta_moda, index_col=0)
#%% ===============================================================================================
# FUNCIONES DEFINIDAS
# =================================================================================================
# Creamos una función para probar distintos árboles de decisión en la clasificación multiclase, 
# variando su profundidad máxima y analizando su exactitud

def resultado() -> dict():
    res = {}
    
    for profundidad in range(1, 11): #profundidades entre 1 y 10
     tree = DecisionTreeClassifier(
        max_depth = profundidad,
        random_state=42
     )
     tree.fit(X_dev, Y_dev)
    
     y_predict = tree.predict(X_heldout)
     score = accuracy_score(Y_heldout, y_predict) * 100
    
     res[profundidad] = score
     print(f'Profundidad: {profundidad}, Exactitud: {score:.2f}%') # print para observar el avance
     
     test_acc = tree.score(X_heldout, Y_heldout)
     test_scores.append(test_acc)
     
    return res 
#%% ===============================================================================================
# SEPARACIÓN DE VARIABLES 
# =================================================================================================
# X variable explicativa: para las imagenes, esto es, los valores de sus pixeles
# Y variable a explicar: para las clases (el tipo de prenda al cual pertenece)
X = fashion.drop('label', axis=1)
Y = fashion['label']

#%% ===============================================================================================
# CLASIFICACIÓN MULTICLASE
# ¿A cuál de las 10 clases corresponde la imagen? 
# =================================================================================================
clases = [
    'Remera', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botita'
]

# Separamos los datos en desarrollo dev (80%) y validación heldout (20%)
X_dev, X_heldout, Y_dev, Y_heldout = train_test_split(
    X, Y,
    test_size=0.2,          
    stratify=Y,
    random_state=20
)
# Luego, dividimos los datos de desarrolo dev, en train y test para búsqueda de hiperparámetros
X_train, X_test, Y_train, Y_test = train_test_split(
    X_dev, Y_dev, test_size=0.2, stratify=Y_dev, random_state=42
)


#%% Ajustamos un modelo de arbol de decisión y lo entrenamos con distintas profundidades del 1 al 10

test_scores = []
resultados = resultado() #acá utilizamos la función definida al comienzo del archivo

#%% Gráfico Rendimiento vs Profundidad del Árbol

profundidades = list(resultados.keys())
exactitudes = list(resultados.values())

plt.figure(figsize=(8, 5))

plt.plot(profundidades, exactitudes, 'o-', color='blue', markersize=8, linewidth=2, label='Exactitud (Dev)')
plt.xlabel('Profundidad del Árbol', fontsize = 13)
plt.ylabel('Exactitud (%)', fontsize= 13)
plt.title('Rendimiento vs Profundidad del Árbol', fontsize= 13)
plt.xticks(range(1, 11))

plt.grid(True)
plt.show()

#%% Búsqueda de hiperparámetros con validación cruzada 
# Max_features lo dejamos fijo en None porque al considerar todos los atributos conseguimos una mayor exactitud
# El criterio lo dejamos por default en gini porque es menos costoso y no disminuye de manera considerable la exactitud

param_grid = {
    'max_depth': [5, 10, 15],          
    'min_samples_split': [2, 5, 10],   
    'min_samples_leaf': [1, 4, 8],     
    'max_features': [None]     
}

tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,                      # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,        
    verbose=1              
    
)

grid_search.fit(X_train,Y_train)

# Resultados de la búsqueda:
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"\nMejores parámetros: {best_params}")
print(f"Mejor accuracy en validación cruzada: {best_score:.4f}")

#%% Evaluación final con conjunto held-out
best_tree = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
best_tree.fit(X_train,Y_train)


# Predecir y evaluar
y_pred = best_tree.predict(X_heldout)
heldout_acc = accuracy_score(Y_heldout,y_pred)
print(f"\nAccuracy en conjunto held-out: {heldout_acc:.4f}")

#%% Matriz de confusión
cm = confusion_matrix(Y_heldout, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)

plt.figure(figsize=(12, 10))
disp.plot(cmap='Blues', values_format='d', xticks_rotation=45)

for text in disp.text_.ravel():
    text.set_fontsize(7)

# Etiquetas
disp.ax_.set_xlabel("Etiqueta Predicha", fontsize=12)
disp.ax_.set_ylabel("Etiqueta Verdadera", fontsize=12)

plt.title('Matriz de Confusión (Conjunto Held-out)')
plt.tight_layout()
plt.show()
#%%
# Reporte detallado: precision, recall y F1
Reporte = classification_report(Y_heldout, y_pred)
print("Reporte de clasificación:\n", classification_report(Y_heldout, y_pred))
#%% Análisis de errores
error_rates = cm / cm.sum(axis=1)[:, np.newaxis]
np.fill_diagonal(error_rates, 0)  # Eliminar aciertos

for i in range(10):
    max_errors = []
    for j in range(10):
        if i != j and error_rates[i, j] > 0.01:  # Filtramos errores significativos
            max_errors.append((i, j, error_rates[i, j]))

max_errors.sort(key=lambda x: x[2], reverse=True)
print("\nPrincipales confusiones:")
for i, j, rate in max_errors[:10]:
    print(f"{clases[i]} → {clases[j]}: {rate:.2%}")

#%% Calculamos presición y recall por clase 

y_pred = best_tree.predict(X_heldout)

precisionScore = precision_score(Y_heldout, y_pred, average=None)
recallScore = recall_score(Y_heldout, y_pred, average=None)

#%% Gráfico que visualiza la precisión y recall por clase  

clases = list(range(10))  # sabemos que van del 0 al 9
grosor = 0.4

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([c - grosor/2 for c in clases], precisionScore, grosor, label='Precisión', color='lightblue')
ax.bar([c + grosor/2 for c in clases], recallScore, grosor, label='Recall', color='blue')

ax.set_xlabel('Clases', fontsize = 19)
ax.set_ylabel('Valor alcanzado', fontsize = 19)
ax.set_title('Precision & Recall por Clase', fontsize = 20, pad = 20)
ax.set_xticks(clases)
ax.set_xticklabels(clases, fontsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.8)
ax.legend(fontsize=12, handlelength=2, handleheight=1.5)

plt.tight_layout()
plt.show()






