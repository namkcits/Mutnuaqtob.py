import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import from sklearn import datasets
import from sklearn.model_selection import train_test_split
import from sklearn.preprocessing import StandardScaler
import from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import streamlit as st

# Create a simple QNN circuit
def create_qnn_circuit(qubits, params):
    circuit = cirq.Circuit()
    for i in range(len(qubits)):
        circuit += cirq.ry(params[i]).on(qubits[i])
    return circuit

# Prepare data for the quantum circuit
n_qubits = 2
n_params = n_qubits

# Generate qubits
qubits = [cirq.GridQubit(0, i) for i in range(n_qubits)]

# Prepare parameters
params = sympy.symbols('theta(0:{})'.format(n_qubits))

# Create the quantum circuit
qnn_circuit = create_qnn_circuit(qubits, params)

# Convert the quantum circuit to a TensorFlow layer
qnn_layer = tfq.layers.PQC(qnn_circuit, readout_operators=cirq.PauliString(cirq.Z(qubits[-1])))

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build the QNN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_qubits),
    tf.keras.layers.Reshape((n_qubits, 1)),
    qnn_layer
])

# Compile the model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Create a Streamlit app
def app():
    # Set the title of the app
    st.title('Iris Flower Classifier using Quantum Neural Network')
    
    # Load Iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Data preprocessing
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    
    # Ask the user to input values for the features
    feature_names = iris.feature_names
    features = []
    for name in feature_names:
        feature = st.number_input(name)
        features.append(feature)
    X_user = np.array(features).reshape(1, -1)
    X_user = scaler.transform(X_user)
    
    # Predict the class of the input
    y_user = model.predict(X_user)
    class_names = iris.target_names
    class_idx = np.argmax(y_user, axis=1)[0]
    class_name = class_names[class_idx]
    
    # Display the predicted class
    st.write("Predicted class:", class_name)

# Run the app
if __name__ == '__main__':
    app() 
