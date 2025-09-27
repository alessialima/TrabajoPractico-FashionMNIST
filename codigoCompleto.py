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
import seaborn as sns
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

# Devuelve dos listas: exactitud en train y en test para cada profundidad
def resultados_train_test(X_train, Y_train, X_test, Y_test, max_depths=range(1, 11)):
    train_scores, test_scores = [], []

    for profundidad in max_depths:
        tree = DecisionTreeClassifier(max_depth=profundidad, random_state=42)
        tree.fit(X_train, Y_train)

        # Evaluación en TRAIN
        y_train_pred = tree.predict(X_train)
        train_acc = accuracy_score(Y_train, y_train_pred) * 100
        train_scores.append(train_acc)

        # Evaluación en TEST 
        y_test_pred = tree.predict(X_test)
        test_acc = accuracy_score(Y_test, y_test_pred) * 100
        test_scores.append(test_acc)

        print(f"Prof {profundidad:2d} | Train: {train_acc:5.2f}% | Test: {test_acc:5.2f}%")

    return train_scores, test_scores

#%% ===============================================================================================
# SEPARACIÓN DE VARIABLES 
# =================================================================================================
# X variable explicativa: para las imagenes, esto es, los valores de sus pixeles
# Y variable a explicar: para las clases (el tipo de prenda al cual pertenece)
X = fashion.drop('label', axis=1)
Y = fashion['label']    
#%% ===============================================================================================
# EXPLORACIÓN DE DATOS 
# =================================================================================================
"""
Contamos con 10 prendas que, con ayuda del github de Fashion-MNIST, 
podemos observar a qué tipo de prenda pertenece cada número de clase. 
0 = Remera
1 = Pantalón
2 = Suéter
3 = Vestido
4 = Abrigo
5 = Sandalia
6 = Camisa
7 = Zapatillas
8 = Bolsa
9 = Botita
"""
clases = [
    'Remera', 'Pantalón', 'Suéter', 'Vestido', 'Abrigo',
    'Sandalia', 'Camisa', 'Zapatilla', 'Bolso', 'Botita'
]


# Veamos que hay una misma cantidad de prendas por clase 
cantidadPrendasPorClase = fashion['label'].value_counts() 
print(cantidadPrendasPorClase)

# Gráfico que visualiza la comparación de prendas por clase: 
plt.figure(figsize=(12, 6))
unique, counts = np.unique(Y, return_counts=True)
plt.bar([clases[i] for i in unique], counts)
plt.title("Cantidad Imagenes por Clase de Fashion MNIST", fontsize = 14)
plt.xlabel("Prendas de Ropa", fontsize= 14)
plt.ylabel("Cantidad", fontsize= 14)
plt.show()
# obs: se puede observar que hay 7000 imágenes por prenda

#%% Imágenes promedio por clase
# Para buscar un patrón que describa cada clase, realizamos un promedio de intensidad de píxeles de cada una de ellas
plt.figure(figsize=(15, 8))

# Busco valor mínimo y máximo de todas las clases 
all_means = [np.mean(X[Y == i], axis=0).to_numpy() for i in range(10) if np.sum(Y == i) > 0]
vmin, vmax = np.min(all_means), np.max(all_means)

for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    
    if np.sum(label) > 0:
        img = np.mean(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='Reds', vmin=vmin, vmax=vmax)  # Misma escala para todos
        plt.title(clases[i], fontsize=20)
        plt.axis('off')

plt.subplots_adjust(wspace=0.08, hspace=0.08, right=0.9) 

# Colorbar 
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=18)

plt.suptitle('Imágenes Promedio por Clase', fontsize=28, y=0.95)
plt.show()

#%% Imágenes desviación estándar por clase
# Para analizar la diferencia entre prendas de una misma clase, analizamos la desviación estándar de intesidad de píxeles de cada una de ellas 
plt.figure(figsize=(15, 8))

# Calculamos el rango global de desviaciones estándar 
all_stds = [np.std(X[Y == i], axis=0).to_numpy() for i in range(10) if np.sum(Y == i) > 0]
vmin, vmax = np.min(all_stds), np.max(all_stds)

for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    if np.sum(label) > 0:
        img = np.std(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='bwr', vmin=vmin, vmax=vmax) 
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')

plt.subplots_adjust(wspace=0.08, hspace=0.08, right=0.9)  

# Colorbar 
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.suptitle('Desviación Estándar por Clase', fontsize=28, y=0.95)
plt.show()

#%% Diferencia Promedio por Clases 
# Promedio de píxeles por clase
sueter = fashion[fashion["label"] == 2].iloc[:, :784].mean()
camisa = fashion[fashion["label"] == 6].iloc[:, :784].mean()

# Mostrar diferencia promedio entre clases
diferencia = (sueter - camisa).values.reshape(28, 28)
plt.imshow(diferencia, cmap='bwr', vmin=-150, vmax=150)
plt.colorbar()
plt.title("Diferencia promedio (sueter - camisa)")
ticks = np.arange(0, 28, 2)
plt.xticks(ticks, ticks, rotation=45, fontsize=8)
plt.yticks(ticks, ticks, fontsize=8)
plt.gca().grid(False) 

plt.tight_layout()
plt.show()
#%% ===============================================================================================
# CLASIFICACIÓN BINARIA
# ¿La imagen corresponde a la clase 0 o a la clase 8? 
# =================================================================================================

#%% A partir del dataframe original, construimos uno nuevo que contenga sólo al subconjunto de imágenes que corresponden a la clase 0 y clase 8

# Subconjunto clases 0 y 8
subconjunto_0_8 = duckdb.sql("""
                SELECT *
                FROM fashion
                WHERE label = 0 OR label = 8;
                """).df()
# Este subconjunto está balanceado, se tienen 7000 muestras de cada clase (según lo analizado en el punto 1)
# Separamos el 80% para train y el restante 20% para test, ambos cantidades pares así que hacemos mitad de cada clase
df_0 = duckdb.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 0
    LIMIT 5600
    """).df()

df_8 = duckdb.sql("""
    SELECT *
    FROM subconjunto_0_8
    WHERE label = 8
    LIMIT 5600
    """).df()

subconjunto_0_8_TRAIN = pd.concat([df_0, df_8]).sample(frac=1, random_state=1)  # barajamos
# df y train tienen las mismas columnas
diferencia = subconjunto_0_8.merge(subconjunto_0_8_TRAIN, how='outer', indicator=True)
subconjunto_0_8_TEST = diferencia[diferencia['_merge'] == 'left_only'].drop(columns=['_merge'])
#%% ===============================================================================================
# SELECCIÓN DE ATRIBUTOS
# =================================================================================================

# Paso 1: calculamos los píxeles promedio de cada clase 
remeras_promedio = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos_promedio = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()

# diferencia promedio entre clases
diferencia = (remeras_promedio - bolsos_promedio).values.reshape(28, 28)

# ordenamos a los índices según cuan cercano esté a una clase 
# en este caso, los valores mínimos serán los más cercanos a clase BOLSO 
# y los valores máximos estarán más cerca de la clase REMERA 
 
flat = diferencia.flatten()
indices_ordenados = np.argsort(flat)

# Más azules (valores mínimos): más cercanos a clase BOLSO 
indices_mas_azules = indices_ordenados[:5] # acá pongo cuántos quiero seleccionar 
coordenadas_azules = [divmod(idx, 28) for idx in indices_mas_azules] 

# Más rojos (valores máximos): más cercanos a clase REMERA
indices_mas_rojos = indices_ordenados[-5:] # acá pongo cuántos quiero seleccionar 
coordenadas_rojos = [divmod(idx, 28) for idx in indices_mas_rojos]

# Convertir coordenadas a enteros
coordenadas_azules = [(int(x), int(y)) for x, y in coordenadas_azules]
coordenadas_rojos = [(int(x), int(y)) for x, y in coordenadas_rojos]

print("Pixeles más azules:", indices_mas_azules, "\n", "· Coordenadas:", coordenadas_azules)
print("Pixeles más rojos:", indices_mas_rojos, "\n", "· Coordenadas:",  coordenadas_rojos)

# indices: píxeles 
# coordenadas: sus coordenadas dentro de la matriz 

#%%
conjunto1 = []
for i in range(2):
    conjunto1.append(int(indices_mas_azules[i]))
for j in range(2):
    conjunto1.append(int(indices_mas_rojos[j]))
    # creo el primer combo con los mejores 4 píxeles
print("combo 1", conjunto1)   

conjunto2 = []
for i in range(len(indices_mas_azules)):
    conjunto2.append(int(indices_mas_azules[i]))
for j in range(len(indices_mas_rojos)):
    conjunto2.append(int(indices_mas_rojos[j]))
    # creo el segundo combo con más píxeles 
print("combo 2", conjunto2)   

random.seed(42)
# números random 
num = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
a = num
num2 = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
b = num2
num3 = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
c = num3

# conjunto random de 10 píxeles: 
conjunto3 = []
random.seed(40)
for i in range(8):
    p = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
    conjunto3.append(p)

print("combo3",conjunto3)    

# creamos combinaciones junto con los dos conjuntos y números random 
combinaciones = { 'conjunto 1': conjunto1, 
                 'conjunto 2': conjunto2, 
                 'conjunto 3': conjunto3
                } 
combinaciones_lista = list(combinaciones.keys())

print("numeros random",a,b,c) 

combinaciones2 = {'conjunto 2 & nro random': conjunto2 + [a,b,c],
                  'conjunto unión': conjunto2 + conjunto3
    }
combinaciones_lista2 = list(combinaciones2.keys())
#%% Evaluación / métricas 

valores_k = range(1, 20)
resultados = []

# Etiquetas
y_train = subconjunto_0_8_TRAIN["label"].values
y_test = subconjunto_0_8_TEST["label"].values

todas_las_combinaciones = combinaciones | combinaciones2
resultados = [] 

for nombre, atributos in todas_las_combinaciones.items():
    X_train = subconjunto_0_8_TRAIN.iloc[:, atributos].values
    X_test = subconjunto_0_8_TEST.iloc[:, atributos].values

    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)

        y_train_pred = modelo.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        resultados.append({
            "combinacion": nombre,
            "atributos": atributos,
            "k": k,
            "test_accuracy": acc * 100,
            "train_accuracy": train_acc * 100,
            "precision": report['macro avg']['precision'] * 100,
            "recall": report['macro avg']['recall'] * 100,
            "f1": report['macro avg']['f1-score'] * 100
        })

# Resultados:
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

#%% MEJORES K PARA CADA COMBINACIÓN 
mejores_k_por_combinacion = df_resultados.loc[df_resultados.groupby('combinacion')['test_accuracy'].idxmax()]
mejores_k_ordenados = mejores_k_por_combinacion.sort_values('test_accuracy', ascending=False)

print(mejores_k_ordenados)
#%% Números random utilizados en combo 3 
print("\nNumeros random:", a, b, c) 

#%% Gráfico exactitud por k segun mejores combos 
colores = {
    'conjunto 1': 'tab:blue',
    'conjunto 2': 'tab:orange',
    'conjunto 3': 'tab:green',
    'conjunto 2 & nro random': 'tab:red',
    'conjunto unión': 'tab:purple'
}

# GRUPO BÁSICO
plt.figure(figsize=(12, 7))
for nombre in combinaciones_lista:
    subset = df_resultados[df_resultados['combinacion'] == nombre]
    plt.plot(subset['k'], subset['train_accuracy'], label=f'{nombre} - Train', linestyle='--', color=colores[nombre])
    plt.plot(subset['k'], subset['test_accuracy'], label=f'{nombre} - Test', marker='o', color=colores[nombre])

plt.title('Accuracy en entrenamiento vs test según k (conjuntos 1, 2 y 3)', fontsize=20)
plt.xlabel('Vecinos más cercanos (k)', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)
plt.xticks(valores_k)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()

# GRUPO mezclas
plt.figure(figsize=(12, 7))
for nombre in combinaciones_lista2:
    subset = df_resultados[df_resultados['combinacion'] == nombre]
    plt.plot(subset['k'], subset['train_accuracy'], label=f'{nombre} - Train', linestyle='--', color=colores[nombre])
    plt.plot(subset['k'], subset['test_accuracy'], label=f'{nombre} - Test', marker='o', color=colores[nombre])

plt.title('Accuracy en entrenamiento vs test según k (conjuntos mejorados)', fontsize=20)
plt.xlabel('Vecinos más cercanos (k)', fontsize=20)
plt.ylabel('Accuracy (%)', fontsize=20)
plt.xticks(valores_k)
plt.grid(True)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()

#%% Matriz de Confusión de los mejores 5 casos

# Filtramos valores de k entre 4 y 20, evitando sobreajuste 
k_validos = list(range(4, 20))
df_filtrado = df_resultados[df_resultados['k'].isin(k_validos)]

# Los mejores dentro de ese rango
mejores_k_restringidos = df_filtrado.loc[df_filtrado.groupby('combinacion')['test_accuracy'].idxmax()]
mejores_k_restringidos = mejores_k_restringidos.sort_values('test_accuracy', ascending=False)

combos_seleccionados = mejores_k_restringidos['atributos'].tolist()
titulos_matrices = mejores_k_restringidos['combinacion'].tolist()
k_seleccionadas = mejores_k_restringidos['k'].tolist()

for i in range(len(combos_seleccionados)): 
    pixeles_seleccionados = combos_seleccionados[i]
    k = k_seleccionadas[i]
    X_train = subconjunto_0_8_TRAIN.iloc[:, pixeles_seleccionados].values
    X_test = subconjunto_0_8_TEST.iloc[:, pixeles_seleccionados].values

    modelo = KNeighborsClassifier(n_neighbors = k) 
    modelo.fit(X_train, y_train) 
    y_pred = modelo.predict(X_test) 

# Calculo Exactitud 
    accuracy = accuracy_score(y_test, y_pred) 
    print("Mejores Casos","\n",f"Combo: {i+1}",f"Accuracy: {accuracy:.3f}") 

# Gráfico de la matriz de confusión: 
    y_pred = modelo.predict(X_test) 

    cmKNN = confusion_matrix(y_test, y_pred) 
    dispKNN = ConfusionMatrixDisplay(confusion_matrix = cmKNN, display_labels = [0,8])

    plt.figure(figsize=(2.5, 2.5)) 
    dispKNN.plot(cmap='Blues', values_format='d', colorbar = False, ax=plt.gca()) 
    plt.gca().set_aspect('equal', 'box')
    plt.gca().grid(False) 

    dispKNN.ax_.set_xlabel("Predicted", fontsize = 12) 
    dispKNN.ax_.set_ylabel("Actual", fontsize = 12) 
    dispKNN.ax_.set_title(titulos_matrices[i] + ' con k = ' + str(k_seleccionadas[i]), fontsize = 12) 

    plt.tight_layout()
    plt.show()


#%% Gráfico que visualiza la comparación entre los promedios de ambas clases
# además de mostrar en los extremos el promedio de cada clase en sí
plt.figure(figsize=(15, 5))    
plt.subplot(1, 3, 1)
label0 = (Y == 0)
img0 = np.mean(X[label0], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img0, cmap='Reds')
plt.colorbar(fraction=0.046, pad=0.04)  
plt.gca().grid(False) 

plt.subplot(1, 3, 3)
label8 = (Y == 8)
img8 = np.mean(X[label8], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img8, cmap='Blues')
plt.colorbar(fraction=0.046, pad=0.04)  
plt.gca().grid(False) 

plt.subplot(1,3,2)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (bolsos-remeras).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='RdBu')


cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(),diferencia.max()])
cbar.set_ticklabels(['REMERA','BOLSO']) 
cbar.ax.tick_params(labelsize=10)


plt.subplots_adjust(wspace=0.3, right=0.85) 

plt.show()

#%% VISUALIZACIÓN DE POSICIÓN DE LOS PÍXELES QUE USAREMOS EN CADA CONJUNTO 

#paso 1: armo coordenadas con los pixeles 
coordenadas_conj3 = [divmod(idx, 28) for idx in conjunto3]
coordenadas_conj2 = [divmod(idx, 28) for idx in conjunto2]    
coordenadas_conj1 = [divmod(idx, 28) for idx in conjunto1]

# Conjunto 1 
plt.figure(figsize=(15, 5))  
plt.subplot(1,3,1)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (bolsos-remeras).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='RdBu')
plt.title("Posición de píxeles conjunto 1")
for (fila, columna) in coordenadas_conj1:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2) 
plt.gca().grid(False) 

cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(),diferencia.max()])
cbar.set_ticklabels(['REMERA','BOLSO']) 
cbar.ax.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.3, right=0.85) 

# Conjunto 2
plt.subplot(1,3,2)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (bolsos-remeras).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='RdBu')
plt.title("Posición de píxeles conjunto 2")
for (fila, columna) in coordenadas_conj2:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2) 
plt.gca().grid(False) 

cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(),diferencia.max()])
cbar.set_ticklabels(['REMERA','BOLSO']) 
cbar.ax.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.3, right=0.85) 

# Conjunto 3 
plt.subplot(1,3,3)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (bolsos-remeras).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='RdBu')
plt.title("Posición de píxeles conjunto 3")
for (fila, columna) in coordenadas_conj3:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2) 
plt.gca().grid(False) 

cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(),diferencia.max()])
cbar.set_ticklabels(['REMERA','BOLSO']) 
cbar.ax.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.3, right=0.85) 

plt.show()

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


#%% Ajustamos un árbol de decisión con profundidades 1‑10 y medimos exactitud en TRAIN y TEST
train_scores, test_scores = resultados_train_test(
    X_train, Y_train, X_test, Y_test, max_depths=range(1, 11)
)

#%%

#%% Gráfico comparando caso train vs test, exactitud en función de profundidad 
profundidades = list(range(1, 11))

plt.figure(figsize=(9, 5))
plt.plot(profundidades, train_scores, 'o-', label='Train', linewidth=2, markersize=7)
plt.plot(profundidades, test_scores,  's-', label='Test',  linewidth=2, markersize=7)

plt.xlabel('Profundidad del Árbol', fontsize=16)
plt.ylabel('Exactitud (%)', fontsize=16)
plt.title('Exactitud en Train vs Test según la Profundidad', fontsize=18)
plt.xticks(profundidades)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.show()

#%% Búsqueda de hiperparámetros con validación cruzada 
# El criterio lo dejamos por default en gini porque es menos costoso y no disminuye de manera considerable la exactitud

param_grid = {
    'max_depth': [5, 7, 10],          
    'min_samples_split': [5, 10, 15],   
    'min_samples_leaf': [2, 4],     
    'max_features': ['sqrt', 'log2', 0.5]    # sacamos la opción None de acá para que aparezca como mejor resultado nuestro mf seleccionado 
}

tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    cv=5,                     
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
# Elegimos como mejor caso profundidad 10, mf: 0.5, mss: 10, msl: 4

#%%

#%% Evaluamos sobre los distintos conjuntos para comprobar que no haya sobreajuste
# Como quitamos la opción None, este código nos devolverá la exactitud en dev y held-out del caso con profundidad 10, mf: 0.5, mss: 10, msl: 4

best_tree = grid_search.best_estimator_

metrics = {}

# TRAIN
y_pred_train = best_tree.predict(X_train)
metrics['train_acc'] = accuracy_score(Y_train, y_pred_train)

# TEST 
y_pred_test = best_tree.predict(X_test)
metrics['test_acc'] = accuracy_score(Y_test, y_pred_test)

# HELDOUT
y_pred_hold = best_tree.predict(X_heldout)
metrics['heldout_acc'] = accuracy_score(Y_heldout, y_pred_hold)

print("\n====== Exactitud ======")
for split, acc in metrics.items():
    print(f"{split:10s}: {acc*100:.2f}%")

#%% Gráfico de variación de cada mf en el caso más óptimo de las profundidades elegidas. Considera únicamente el mejor mss y msl en cada punto
# Armo un dataframe con los resultados para cada mf distinto 
results_df = pd.DataFrame(grid_search.cv_results_)
results_df['param_max_features'] = (
    results_df['param_max_features']
    .replace({None: "None", 0.5: "50%"})
)
# Gráfico para cada mf, exactitud en función de profundidad 
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Colores 

# Graficamos para cada valor de max_features
for i, mf in enumerate(results_df['param_max_features'].unique()):
    # Filtramos los datos para este max_features específico
    mf_data = results_df[results_df['param_max_features'] == mf]
    
    # Agrupamos por profundidad y obtenemos el mejor accuracy 
    grouped = mf_data.groupby('param_max_depth')['mean_test_score'].max()
    
    
    plt.plot(grouped.index, grouped.values, 
             color=colors[i],
             marker='o',
             linestyle='-',
             linewidth=2,
             markersize=8,
             label=f'max_features={mf}')
   
# Personalizamos el gráfico
plt.title('Exactitud vs. Profundidad para diferentes mf', fontsize=20, pad=20)
plt.xlabel('Profundidad máxima del árbol (max_depth)', fontsize=18)
plt.ylabel('Exactitud (%)', fontsize=18)
plt.xticks(param_grid['max_depth'], fontsize=17)
plt.yticks(fontsize=17)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=16)
plt.tight_layout()


plt.show()

#%% Matriz de confusión en nuestro caso elegido 
cm = confusion_matrix(Y_heldout, y_pred_hold)

# Generar figura más compacta
fig, ax = plt.subplots(figsize=(6,5))

# Mostrar matriz con etiquetas personalizadas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)

# Ajustar texto dentro de las celdas
for text in disp.text_.ravel():
    text.set_fontsize(8)

# Etiquetas y título
ax.set_xlabel("Etiqueta Predicha", fontsize=11)
ax.set_ylabel("Etiqueta Verdadera", fontsize=11)
ax.set_title("Matriz de Confusión (Held-out)", fontsize=13)

# Ajuste del tamaño de los ticks
ax.tick_params(axis='x', labelsize=9)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.gca().grid(False) 
plt.show()

#%%
# Reporte detallado: precision, recall y F1
print("\nReporte de clasificación (held‑out):")
print(classification_report(Y_heldout, y_pred_hold,
                            target_names=clases, digits=3))
#%% Análisis de errores

row_sums = cm.sum(axis=1, keepdims=True)
error_rates = np.divide(cm, row_sums, where=row_sums != 0)
np.fill_diagonal(error_rates, 0)     # ponemos 0 en los aciertos

# Extraemos confusiones >1 %
confusions = [
    (i, j, error_rates[i, j])
    for i in range(len(clases))
    for j in range(len(clases))
    if i != j and error_rates[i, j] > 0.01
]
confusions.sort(key=lambda x: x[2], reverse=True)

print("\nPrincipales confusiones (>1 % de la clase):")
for i, j, rate in confusions[:10]:
    print(f"{clases[i]} → {clases[j]}: {rate:.2%}")
#%% Calculamos presición y recall por clase 

prec  = precision_score(Y_heldout, y_pred_hold, average=None)
rec   = recall_score(Y_heldout, y_pred_hold, average=None)

#%% Gráfico que visualiza la precisión y recall por clase  

class_ids = list(range(10))      # 0‑9 para posiciones en el eje X
grosor = 0.4

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([c - grosor/2 for c in class_ids], prec, grosor,
       label='Precisión', color='lightblue')
ax.bar([c + grosor/2 for c in class_ids], rec,  grosor,
       label='Recall',    color='blue')

ax.set_xlabel('Clases', fontsize=19)
ax.set_ylabel('Valor alcanzado', fontsize=19)
ax.set_title('Precisión & Recall por Clase', fontsize=20, pad=20)

ax.set_xticks(class_ids)
ax.set_xticklabels(class_ids, fontsize=14)   # si querés que muestre 0‑9
# ó  ax.set_xticklabels(clases, fontsize=14, rotation=45)  # si preferís los nombres reales

ax.tick_params(axis='y', labelsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.8)
ax.legend(fontsize=12, handlelength=2, handleheight=1.5)

plt.tight_layout()
plt.show()
























