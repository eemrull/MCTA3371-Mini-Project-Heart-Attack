import sys
import random
import matplotlib
import pandas as pd
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import mean_squared_error

# Genetic Algorithm parameters
population_size = 50
generations = 100
mutation_rate = 0.03
crossover_rate = 1

# Fuzzy Logic parameters (dummy values, replace with your logic)
fuzzy_input_threshold = 0.5
fuzzy_output_threshold = 0.7

# Given items
items = [
    ("Clothes", 2, 10),
    ("Laptop", 3, 8),
    ("Sunglasses", 0.8, 5),
    ("Hat", 0.7, 6),
    ("Travel Magazine", 0.75, 4),
    ("Power Bank", 1.5, 5.5),
    ("Book", 1.1, 4.5),
    ("Snacks", 0.7, 5),
    ("Cosmetics", 1, 8)
]

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # Create input boxes for parameters
        self.population_size_input = QtWidgets.QLineEdit(self)
        self.generations_input = QtWidgets.QLineEdit(self)
        self.mutation_rate_input = QtWidgets.QLineEdit(self)
        self.crossover_rate_input = QtWidgets.QLineEdit(self)

        # Create labels for input boxes
        population_label = QtWidgets.QLabel("Population Size:")
        generations_label = QtWidgets.QLabel("Generations:")
        mutation_rate_label = QtWidgets.QLabel("Mutation Rate:")
        crossover_rate_label = QtWidgets.QLabel("Crossover Rate:")

        # Create the Matplotlib FigureCanvas object.
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)

        # Plot an initial empty chart
        self.sc.axes.plot([], [])
        self.sc.draw()

        # Create "Run Genetic Algorithm" button
        run_button = QtWidgets.QPushButton("Run Genetic Algorithm", self)
        run_button.clicked.connect(self.run_genetic_algorithm)

        # Create layout for input boxes and button
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

        # Create a widget to hold the layout
        input_widget = QtWidgets.QWidget()
        input_widget.setLayout(layout)

        # Create a central layout that contains both the Matplotlib canvas and the input widget
        central_layout = QtWidgets.QHBoxLayout()
        central_layout.addWidget(input_widget)
        central_layout.addWidget(self.sc)
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(central_layout)

        self.setCentralWidget(central_widget)
        self.show()

    def run_genetic_algorithm(self):
        # Your existing logic for running the genetic algorithm
        population_size = int(self.population_size_input.text())
        generations = int(self.generations_input.text())
        mutation_rate = float(self.mutation_rate_input.text())
        crossover_rate = float(self.crossover_rate_input.text())

        # Run the genetic algorithm with the specified parameters
        genetic_algorithm_output = self.genetic_algorithm(population_size, generations, mutation_rate, crossover_rate)

        # Run fuzzy logic system with the genetic algorithm output
        fuzzy_output = self.fuzzy_logic_system(genetic_algorithm_output['population'],
                                               genetic_algorithm_output['fitness_scores'])

        # Combine genetic algorithm and fuzzy logic results
        combined_output = self.combine_genetic_and_fuzzy(genetic_algorithm_output, fuzzy_output)

        # Plot the final results
        self.plot_results(combined_output)

    def genetic_algorithm(self, population_size, generations, mutation_rate, crossover_rate):
        # Your genetic algorithm logic here
        # ...

        # Example: A dummy genetic algorithm that returns random results
        population = [tuple(random.choices([0, 1], k=len(items))) for _ in range(population_size)]
        fitness_scores = [random.random() for _ in range(population_size)]

        return {'population': population, 'fitness_scores': fitness_scores}

    def fuzzy_logic_system(self, population, fitness_scores):
        # Your fuzzy logic system logic here
        # ...

        # Example: A dummy fuzzy logic system that returns random results
        fuzzy_input = [random.random() for _ in range(len(population))]
        fuzzy_output = [random.random() for _ in range(len(population))]

        return {'fuzzy_input': fuzzy_input, 'fuzzy_output': fuzzy_output}

    def combine_genetic_and_fuzzy(self, genetic_output, fuzzy_output):
        # Your combination logic here
        # ...

        # Example: A dummy combination logic that uses mean squared error
        combined_output = mean_squared_error(genetic_output['fitness_scores'], fuzzy_output['fuzzy_output'])

        return combined_output

    def plot_results(self, combined_output):
        # Your plotting logic here
        # ...

        # Example: Plot the combined output (replace with your actual plotting logic)
        self.sc.axes.clear()
        self.sc.axes.plot([0, 1], [combined_output, combined_output], label='Combined Output')
        self.sc.axes.legend()
        self.sc.draw()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())
