import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


#Importamos datos precargados
boston=datasets.load_boston()
print(boston)
print()

#Verificamos la información contenida en el Dataset
print('Información en el Dataset:')
print(boston.keys())
print()

#Verificamos las caracteristicas del Dataset
print('Caracteristicas del Dataset:')
print(boston.DESCR)

#Verificar la cantidad de datos del Dataset
print('Cantidad de datos:')
print(boston.data.shape)
print()

#Verificar la información de las columnas
print('Nombres columnas:')
print(boston.feature_names)

#Se selecciona la columna 6 del Dataset
X_adr=boston.data[:,np.newaxis,5]

#Datos de Y
Y_adr=boston.target

#Se grafican los datos seleccionados
plt.scatter(X_adr,Y_adr)
plt.show()

#Se separan los datos
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_adr,Y_adr,test_size=0.2)

#Se importa el algoritmo
from sklearn.tree import DecisionTreeRegressor
adr=DecisionTreeRegressor(max_depth=5)

#Se ejecuta el entrenamiento del modelo
adr.fit(X_train,Y_train)

#Realizo una predicción
Y_pred=adr.predict(X_test)


#Graficamos los datos junto al modelo
X_grid=np.arange(min(X_test),max(X_test),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X_test, Y_test)
plt.plot(X_grid, adr.predict(X_grid), color='red',linewidth=3)
plt.show()


print()
print('Datos del modelo de regresión arboles de decisión')
print()

print('Precisión del modelo:')
print(adr.score(X_train,Y_train))


