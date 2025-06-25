import numpy as np
import pandas as pd

class PerceptronSimple:
    def __init__(self, input_size, learning_rate , epochs ):
        self.weights = np.zeros(input_size + 1)
        self.lr = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return  1 if x>=0 else 0

    def predict(self,x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return self.activation_function(z)

    def train(self, x, y):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch +1}:")
            for xi,target in zip(x,y):
                output = self.predict(xi)
                error = target - output
                self.weights[1:] += self.lr * error * xi # Ajuste de pesos de las entradas
                self.weights[0] += self.lr * error #Ajuste del bias o sesgo
                print(f"  Entrada: {xi}, Esperado: {target}, Predicho: {output}, Error: {error}")

#Ciclo principal
if __name__ == "__main__":
    #Leer archivo Excel
    archivo_excel = pd.read_excel('Recursos/CompuertaAND.xlsx')
    #Extraer solo las columnas X1, X2, X3, X4
    X = archivo_excel[["X1", "X2", "X3", "X4"]].values
    Y = archivo_excel["d"].values

    #Crear y entrenar el Perceptron
    perceptronAND = PerceptronSimple(4, 0.001, 15)
    perceptronAND.train(X, Y)

# Pruebas
print("\nPruebas finales:")
for xi in X:
    print(f"Entrada: {xi}, Predicción: {perceptronAND.predict(xi)}")


