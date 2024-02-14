import sys
import random
import matplotlib
import pandas as pd
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential

population_size = 50
generations = 100
mutation_rate = 0.03
crossover_rate = 1

fuzzy_input_threshold = 0.5
fuzzy_output_threshold = 0.7

items = [
    ("heart", 2, 10)
]

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.population_size_input = QtWidgets.QLineEdit(self)
        self.generations_input = QtWidgets.QLineEdit(self)
        self.mutation_rate_input = QtWidgets.QLineEdit(self)
        self.crossover_rate_input = QtWidgets.QLineEdit(self)

        population_label = QtWidgets.QLabel("Population Size:")
        generations_label = QtWidgets.QLabel("Generations:")
        mutation_rate_label = QtWidgets.QLabel("Mutation Rate:")
        crossover_rate_label = QtWidgets.QLabel("Crossover Rate:")

        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes.plot([], [])
        self.sc.draw()

        # run NN Button
        run_button = QtWidgets.QPushButton("Run Neural Network", self)
        run_button.clicked.connect(self.run_neural_network)

        # layouts
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(population_label)
        layout.addWidget(self.population_size_input)
        layout.addWidget(generations_label)
        layout.addWidget(self.generations_input)
        layout.addWidget(mutation_rate_label)
        layout.addWidget(self.mutation_rate_input)
        layout.addWidget(crossover_rate_label)
        layout.addWidget(self.crossover_rate_input)
        layout.addWidget(run_button)

        input_widget = QtWidgets.QWidget()
        input_widget.setLayout(layout)

        central_layout = QtWidgets.QHBoxLayout()
        central_layout.addWidget(input_widget)
        central_layout.addWidget(self.sc)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)
        self.show()

    def run_neural_network(self):
        # input values
        population_size = int(self.population_size_input.text())
        generations = int(self.generations_input.text())
        mutation_rate = float(self.mutation_rate_input.text())
        crossover_rate = float(self.crossover_rate_input.text())

        # training nn
        mse = self.train_neural_network()

        # plot
        self.plot_results(mse)

    def train_neural_network(self):
        # dummy data 
        X_train = tf.constant([[1, 2], [2, 3], [3, 4]])
        y_train = tf.constant([1, 2, 3])

        # build simple model
        model = Sequential([
            Dense(10, activation='relu', input_shape=(2,)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        # train
        model.fit(X_train, y_train, epochs=10, verbose=0)

        # calculate mean squared error
        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)

        return mse

    def plot_results(self, mse):
        # plot mse
        self.sc.axes.clear()
        self.sc.axes.plot([0, 1], [mse, mse], label='Mean Squared Error')
        self.sc.axes.legend()
        self.sc.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
