import numpy as np
import pandas as pd
# Función de activación: sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la sigmoide
def sigmoid_derivative(x):
    return x * (1 - x)

#Leer archivo Excel
archivo_excel = pd.read_excel('Recursos/CompuertaXOR.xlsx') 
#Extraer solo las columnas X1, X2, X3, X4
X = archivo_excel[["X1", "X2", "X3", "X4"]].values
y = archivo_excel[["d"]].values

# Inicialización
np.random.seed(42)
input_size = 4
hidden_size = 8
output_size = 1
learning_rate = 0.5
epochs = 1000

# Pesos y sesgos
weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
bias_output = np.zeros((1, output_size))

# Entrenamiento
for epoch in range(epochs):
    # Forward
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)

    # Backpropagation
    output_error = y - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Actualizar pesos y sesgos
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(output_error))
        print(f"Época {epoch}, Pérdida: {loss:.4f}")

# Pruebas finales
print("\nPruebas:")
for xi in X:
    hidden = sigmoid(np.dot(xi, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
    print(f"Entrada: {xi}, Salida: {output.round(3)}")
