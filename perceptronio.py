# Módulos importados
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math

# Implementación del perceptrón

# Crear una o varias variables para los pesos de las entradas dependiendo del número de entradas.
# Deben tener valores iniciales aleatorios entre 0 y 1.
inputs = np.array([[0,1], [0,0], [1,1], [1,0], [0,1] ,[1,1]]) #array para probar ejercicio 1
class Perceptronio:
    def __init__(self):
        self.pesos = [] #inicializamos los pesos
    def gen_pesos(self):
        self.pesos = np.random.choice([0., 1.], size=((1 + inputs.shape[1]),), p=[.5, .5]) #Se generan pesos entre 0 y 1 random
    def predict(self, inputs):
        z = np.dot(inputs, self.pesos[1:]) + self.pesos[0] # función dot para obtener la suma de xi.wi
        #print(z), se comenta por que sale un cuadro enorme de texto con los valores ya aplicando en entrenamiento
        phi = np.where(z >= 0.0, 1, 0)# se seleccionan de
        return phi

perceptronio = Perceptronio() #Se crea instancia
perceptronio.gen_pesos() #generar datos muestra
#prueba de funcionamiento
phi = perceptronio.predict(inputs) # demo de ejecución
print (phi)

perceptronio = Perceptronio()
perceptronio.gen_pesos()
#prueba de funcionamiento
phi = perceptronio.predict(inputs)
# Base de datos Iris
iris_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Selección de Setosa (0) y Versicolor (1)
data_y = iris_data.iloc[0:100, 4].values
data_y = np.where(data_y == 'Iris-setosa', 0, 1)

# Selección de longitud de sépalo y pétalo
data_x = iris_data.iloc[0:100, [0,2]].values
#print(data_x)
#print(data_x[1:])
# Gráfica de los datos
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
data_color = ['r' if i == 0 else 'b' for i in data_y]
plt.scatter(data_x[:,0],data_x[:,1],c=data_color, marker='d', alpha=0.5)
#plt.show()
#E2
# Implementar la función de SGD para calcular nuevos pesos a las entradas tomando en cuenta un conjunto de datos de entrada con sus respectivas
# salidas. Además de estos valores de entrada, recibe un valor para la taza de aprendizaje

#se agrega una propiedad para que tome en cuenta la tasa
def get_learning_rate(self):
        return self.learning_rate
def set_learning_rate(self, value):
        self.learning_rate = value
Perceptronio.get_learning_rate = get_learning_rate
Perceptronio.set_learning_rate = set_learning_rate


def sgd(self, inputs, label):
    for i, l in zip(inputs, label):
        delta_w = self.learning_rate * (l - self.predict(i))
        self.pesos[1:] += delta_w * i
        self.pesos[0] += delta_w
    return self

#se extiende la función sgd
Perceptronio.sgd = sgd

# Implementar la función de entrenamiento para el perceptrón, tomando como entrada un conjunto grande de valores de entrada con sus respectivos
# valores de salida. Además recibe el valor para la taza de aprendizaje y un número de 'épocas', las cuales indican cuantas veces se realiza el
# el proceso de entrenamiento.
def get_epochs(self):
        return self.epochs
def set_epochs(self, value):
        self.epochs = value
        
Perceptronio.get_epochs = get_epochs
Perceptronio.set_epochs = set_epochs
#E3
def train(self, train_inputs, train_labels):
    self.pesos = np.zeros(1 + train_inputs.shape[1])
    for _ in range(self.epochs):
        sgd(self, train_inputs, train_labels)
        
    return self
#E4
Perceptronio.train = train
perceptronio = Perceptronio()
perceptronio.learning_rate = 0.1
perceptronio.epochs = 10
perceptronio.train(data_x, data_y)

from matplotlib.colors import ListedColormap
def plotRegions(X, y):
    resolution=0.02
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = perceptronio.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

# Visualización de los datos
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
data_color = ['r' if i == 0 else 'b' for i in data_y]
plt.scatter(data_x[:,0],data_x[:,1],c=data_color, marker='d', alpha=0.5)
plotRegions(data_x, data_y)
plt.show()