from threading import Thread
import time
import numpy as np
from dataclasses import dataclass, field
import pickle
from utils import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BATCH_SIZE = 32


def load_data(filepath: str = 'mnist_train.csv'):
    with open(filepath, 'r') as f:
        f.readline()
        X, y = [], []
        while line := f.readline():
            y_data, *X_data = map(float, line.strip().split(','))
            X.append(X_data)
            y.append(y_data)

    ratio = 0.1
    length = len(X)
    split_slice = int(length * ratio)
    X_test, X_train = np.array(X[:split_slice]), np.array(X[split_slice:])
    y_test, y_train = np.array(y[:split_slice]), np.array(y[split_slice:])

    y_test = y_test.astype(np.int64)
    y_train = y_train.astype(np.int64)
    new_y_test = np.zeros((y_test.size, y_test.max() + 1))
    new_y_test[np.arange(y_test.size), y_test] = 1
    new_y_train = np.zeros((y_train.size, y_train.max() + 1))
    new_y_train[np.arange(y_train.size), y_train] = 1
    y_test = new_y_test
    y_train = new_y_train

    return X_test/255., X_train/255., y_test, y_train


class Character:
    def __init__(self) -> None:
        self.nn = [Layer(28*28, 16, sigmoid),
                   Layer(16, 16, ReLU),
                   Layer(16, 16, ReLU),
                   Layer(16, 10, softmax)]

    def forward(self, inputs: np.ndarray):
        output = inputs
        for layer in self.nn:
            output = layer.forward(output)
        return output


@dataclass
class Population:
    size: int
    X: list[np.ndarray] = field(default_factory=[])
    y: list[np.ndarray] = field(default_factory=[])
    generation: int = 0
    best_character: Character = None
    thread_stop: bool = False

    def __post_init__(self) -> None:
        self.population = [Character() for _ in range(self.size)]
        self.X_batches = list(divide_chunks(self.X, BATCH_SIZE))
        self.y_batches = list(divide_chunks(self.y, BATCH_SIZE))
        self.batch_amount = len(self.y_batches)
        self.fitnesses: list[float] = []
        self.plt_avg_fitness: list[float] = []

    def plt_data(self) -> None:
        while not self.thread_stop:
            plt.clf()
            plt.plot(self.plt_avg_fitness)
            plt.pause(0.1)

    @property
    def mutation_rate(self) -> float:
        return 0.9*np.exp(-self.generation/125)+0.1

    def calculate_fitness(self) -> None:
        self.fitnesses = []
        for character in self.population:
            random_index = np.random.randint(0, self.batch_amount)
            result = character.forward(self.X_batches[random_index])
            # error = abs(self.y_batches[random_index] - result)
            fitness = np.multiply(result, self.y_batches[random_index])
            self.fitnesses.append(np.mean(np.max(fitness, axis=1)))
        self.plt_avg_fitness.append(np.mean(self.fitnesses))

    def generate_new_population(self) -> None:
        new_population = []
        self.find_best_character()
        new_population.append(self.best_character)

        parents = np.random.choice(self.population,
                                   self.size*2,
                                   p=self.fitnesses/np.sum(self.fitnesses))
        new_character = Character()
        rand_mask_weights = []
        rand_mask_biases = []
        for new_layer in new_character.nn:
            rand_mask_weights.append(np.random.choice([True, False],
                                                      size=[self.size] + list(new_layer.weights.shape))[0])
            rand_mask_biases.append(np.random.choice([True, False],
                                                     size=[self.size] + list(new_layer.biases.shape))[0])
        while (current_length := len(new_population)) < self.size:
            parent_A: Character = parents[current_length]
            parent_B: Character = parents[current_length+self.size]

            new_character = Character()

            for i, (new_layer, layer_A, layer_B) in enumerate(zip(new_character.nn, parent_A.nn, parent_B.nn)):
                new_layer.weights[rand_mask_weights[i]] = \
                    layer_A.weights[rand_mask_weights[i]].copy()
                new_layer.weights[~rand_mask_weights[i]] = \
                    layer_B.weights[~rand_mask_weights[i]].copy()
                random_array = np.random.random(size=new_layer.weights.shape)
                mutation_slice = random_array < self.mutation_rate
                new_layer.weights[mutation_slice] += np.random.uniform(-0.1, 0.1)

                new_layer.biases[rand_mask_biases[i]] = \
                    layer_A.biases[rand_mask_biases[i]].copy()
                new_layer.biases[~rand_mask_biases[i]] = \
                    layer_B.biases[~rand_mask_biases[i]].copy()
                random_array = np.random.random(size=new_layer.biases.shape)
                mutation_slice = random_array < self.mutation_rate
                new_layer.biases[mutation_slice] += np.random.uniform(-0.1, 0.1)

            new_population.append(new_character)

        del self.population
        self.population = new_population
        self.generation += 1

    def find_best_character(self) -> None:
        choosen_character = self.population[np.argmax(self.fitnesses)]
        self.best_character = Character()
        for best_layer, choosen_layer in zip(self.best_character.nn, choosen_character.nn):
            best_layer.weights = choosen_layer.weights.copy()
            best_layer.biases = choosen_layer.biases.copy()


def main():
    X_test, X_train, y_test, y_train = load_data()
    population = Population(size=1000, X=X_train, y=y_train)

    try:
        thread = Thread(target=population.plt_data)
        thread.start()
        while True:
        # for _ in range(100):
            population.calculate_fitness()
            population.generate_new_population()
            print(f'Generation: {population.generation:5d}',
                  f'Average Fitness: {np.mean(population.fitnesses):0.5f}',
                  f'Best Fitness: {np.max(population.fitnesses):0.5f}',
                  f'Mutation Rate: {population.mutation_rate:0.5f}',
                  sep='     ')
        population.thread_stop = True
        thread.join()

        print('Training stopped')
        population.calculate_fitness()
        population.find_best_character()
        with open('best_character.pickle', 'wb') as f:
            pickle.dump(population.best_character,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        import test
        print('\a')

    except KeyboardInterrupt:
        print('Training stopped')
        population.thread_stop = True
        thread.join()
        population.calculate_fitness()
        population.find_best_character()
        with open('best_character.pickle', 'wb') as f:
            pickle.dump(population.best_character,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        import test
        print('\a')


if __name__ == '__main__':
    main()
    # with cProfile.Profile() as pr:
    #     main()

    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.dump_stats(filename='nn.prof')
