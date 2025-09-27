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
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    
    if np.sum(label) > 0:
        img = np.mean(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='Reds')
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')
       

plt.subplots_adjust(wspace=0.08, hspace=0.08)  
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.suptitle('Imágenes Promedio por Clase', fontsize=28, y=0.95)

plt.show()



#%% Imágenes desviación estándar por clase
# Para analizar la diferencia entre prendas de una misma clase, analizamos la desviación estándar de intesidad de píxeles de cada una de ellas 
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    label = (Y == i)
    if np.sum(label) > 0:
        img = np.std(X[label], axis=0).to_numpy().reshape(28, 28)
        im = plt.imshow(img, cmap='Blues')
        plt.title(clases[i], fontsize=20) 
        plt.axis('off')

plt.subplots_adjust(wspace=0.08, hspace=0.08)  
cax = plt.axes([0.92, 0.15, 0.02, 0.7])
cbar = plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)

plt.suptitle('Imágenes Desviación Estándar por Clase', fontsize=28, y=0.95)
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
indices_mas_azules = indices_ordenados[:10] # acá pongo cuántos quiero seleccionar 
coordenadas_azules = [divmod(idx, 28) for idx in indices_mas_azules] 

# Más rojos (valores máximos): más cercanos a clase REMERA
indices_mas_rojos = indices_ordenados[-10:] # acá pongo cuántos quiero seleccionar 
coordenadas_rojos = [divmod(idx, 28) for idx in indices_mas_rojos]

# Convertir coordenadas a enteros
coordenadas_azules = [(int(x), int(y)) for x, y in coordenadas_azules]
coordenadas_rojos = [(int(x), int(y)) for x, y in coordenadas_rojos]

print("Pixeles más azules:", indices_mas_azules, "\n", "· Coordenadas:", coordenadas_azules)
print("Pixeles más rojos:", indices_mas_rojos, "\n", "· Coordenadas:",  coordenadas_rojos)

# indices: píxeles 
# coordenadas: sus coordenadas dentro de la matriz 

#%%

conjunto1 = [] # diez mejores píxeles 
for i in range(len(indices_mas_azules)):
    conjunto1.append(int(indices_mas_azules[i]))
for j in range(len(indices_mas_rojos)):
    conjunto1.append(int(indices_mas_rojos[j]))
    # creo el segundo combo con más píxeles 
print("conjunto 1", conjunto1)   


# conjunto random de 10 píxeles: 
conjunto2 = []
i=0
while i < 10:
    p = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
    if not p in conjunto1:
      conjunto2.append(p) 
      i+=1 
    
print("conjunto 2",conjunto2)    

# conjunto random de 15 píxeles: 
conjunto3 = []
i=0
while i < 15:
    p = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
    if not p in conjunto1:
      conjunto3.append(p) 
      i+=1 
    
print("conjunto 3",conjunto3)    

conjunto4 = []
i=0
while i < 5:
    p = int(indices_ordenados[random.randint(0,len(indices_ordenados)-1)])
    if not p in conjunto2:
      conjunto4.append(p) 
      i+=1 
for i in range(3):
    conjunto4.append(int(indices_mas_azules[i]))
for j in range(2):
    conjunto4.append(int(indices_mas_rojos[j]))

print("conjunto 4",conjunto4)    



#%%
# creamos combinaciones junto con los dos conjuntos y números random 
combinaciones = { 'conjunto 1': conjunto1, 
                 'conjunto 2': conjunto2, 
                 'conjunto 3': conjunto3, 
                 'conjunto 4': conjunto4
                } # en conjunto 3 fijamos una lista específica, la cual utilizamos en el informe
combinaciones_lista = list(combinaciones.values())


#%%

valores_k = range(1, 20)
resultados = []

# Etiquetas
y_train = subconjunto_0_8_TRAIN["label"].values
y_test = subconjunto_0_8_TEST["label"].values

i = -1

# Evaluación 
for nombre, atributos in combinaciones.items():
    i+=1
    X_train = subconjunto_0_8_TRAIN.iloc[:, atributos].values
    X_test = subconjunto_0_8_TEST.iloc[:, atributos].values

    for k in valores_k:
        modelo = KNeighborsClassifier(n_neighbors=k)
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        resultados.append({
            "combinacion": nombre,
            "atributos": combinaciones_lista[i],
            "k": k,
            "accuracy": acc * 100,
            "precision": report['macro avg']['precision'] * 100,
            "recall": report['macro avg']['recall'] * 100,
            "f1": report['macro avg']['f1-score'] * 100
        })



# Resultados:
df_resultados = pd.DataFrame(resultados)
print(df_resultados)

#%% MEJORES K PARA CADA COMBINACIÓN 
mejores_k_por_combinacion = df_resultados.loc[df_resultados.groupby('combinacion')['accuracy'].idxmax()]
mejores_k_ordenados = mejores_k_por_combinacion.sort_values('accuracy', ascending=False)

print(mejores_k_ordenados)


#%% Gráfico exactitud por k segun mejores combos 


plt.figure(figsize=(12, 7))
sns.lineplot(
    data=df_resultados,
    x='k',
    y='accuracy',
    hue='combinacion',
    marker='o',
    linewidth=2
)
plt.title('Exactitud según k para cada combinación de píxeles',fontsize=18)
plt.xlabel('Vecinos más cercanos (k)',fontsize=16)
plt.ylabel('Exactitud (%)', fontsize=16)
plt.xticks(valores_k)
plt.grid(True)

plt.legend(
    loc='lower right',         
    fontsize=14,               
    framealpha=1,                           
    facecolor='white',         
)

plt.tight_layout()
plt.show()

#%% Matriz de Confusión de los mejores 4 casos

combos_seleccionados = mejores_k_ordenados['atributos'].tolist()
titulos_matrices = mejores_k_ordenados['combinacion'].tolist()
k_seleccionadas = mejores_k_ordenados['k'].tolist()

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
    dispKNN.ax_.set_title(titulos_matrices[i], fontsize = 12) 

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
coordenadas_conj4 = [divmod(idx, 28) for idx in conjunto4]
coordenadas_conj3 = [divmod(idx, 28) for idx in conjunto3]
coordenadas_conj2 = [divmod(idx, 28) for idx in conjunto2]    
coordenadas_conj1 = [divmod(idx, 28) for idx in conjunto1]

# Conjunto 1 
plt.figure(figsize=(15, 5))  
plt.subplot(1,4,1)
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
plt.subplot(1,4,2)
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
plt.subplot(1,4,3)
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

# Conjunto 4
plt.subplot(1,4,4)
remeras = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 0].iloc[:, :784].mean()
bolsos = subconjunto_0_8_TRAIN[subconjunto_0_8_TRAIN["label"] == 8].iloc[:, :784].mean()
diferencia = (bolsos-remeras).values.reshape(28, 28)
im_diff = plt.imshow(diferencia, cmap='RdBu')
plt.title("Posición de píxeles conjunto 4")
for (fila, columna) in coordenadas_conj4:
    plt.scatter(columna, fila, marker='x', color='black', s=100, linewidth=2) 
plt.gca().grid(False) 

cbar = plt.colorbar(im_diff, fraction=0.046, pad=0.04)
cbar.set_ticks([diferencia.min(),diferencia.max()])
cbar.set_ticklabels(['REMERA','BOLSO']) 
cbar.ax.tick_params(labelsize=10)
plt.subplots_adjust(wspace=0.3, right=0.85) 

plt.show()

#%% Gráfico que visualiza sólo la diferencia de promedios de clase 
plt.imshow(diferencia, cmap='bwr', vmin=-150, vmax=150)
plt.colorbar()
plt.title("Diferencia promedio (remera - bolso)")
ticks = np.arange(0, 28, 2)
plt.xticks(ticks, ticks, rotation=45, fontsize=8)
plt.yticks(ticks, ticks, fontsize=8)
plt.gca().grid(False) 

plt.tight_layout()
plt.show()
#%% Gráficos: desviación de pantalón, diferencia promedio pantalón y zapatilla, desviación de camisa
plt.figure(figsize=(15, 5))    
plt.subplot(1, 3, 1)
label0 = (Y == 1)
img0 = np.std(X[label0], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img0, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplot(1, 3, 3)
label8 = (Y == 6)
img8 = np.std(X[label8], axis=0).to_numpy().reshape(28, 28)
plt.imshow(img8, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplot(1,3,2)
zapatilla = fashion[fashion["label"] == 7].iloc[:, :784].mean()
pantalones = fashion[fashion["label"] == 1].iloc[:, :784].mean()
diferencia = (zapatilla - pantalones).values.reshape(28, 28)
plt.imshow(diferencia, cmap='bwr')
plt.colorbar(fraction=0.046, pad=0.04)
plt.gca().grid(False) 

plt.subplots_adjust(wspace=0.3, right=0.85)  
plt.show()
