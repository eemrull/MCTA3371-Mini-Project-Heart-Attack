import numpy as np
from dataclasses import dataclass, field
import pickle
from utils import *
import cProfile
import pstats
import matplotlib.pyplot as plt
from threading import Thread

BATCH_SIZE = 32


class Character:
    def __init__(self) -> None:
        self.nn = [Layer(8, 16, sigmoid),
                   Layer(16, 16, sigmoid),
                   Layer(16, 16, sigmoid),
                   Layer(16, 2, softmax)]

    def forward(self, inputs: np.ndarray):
        output = inputs
        for layer in self.nn:
            output = layer.forward(output)
        return output


@dataclass
class Population:
    size: int
    X: np.ndarray[np.ndarray]
    y: np.ndarray[np.ndarray]
    generation: int = 0
    best_character: Character = None
    thread_stop: bool = False

    def __post_init__(self) -> None:
        self.population = [Character() for _ in range(self.size)]
        self.fitnesses: list[float] = []
        self.plt_avg_fitness: list[float] = []

    def plt_data(self) -> None:
        while not self.thread_stop:
            plt.clf()
            plt.plot(self.plt_avg_fitness)
            plt.pause(1)

    @property
    def mutation_rate(self) -> float:
        return 0.9*np.exp(-self.generation/125)+0.1

    def calculate_fitness(self) -> None:
        self.fitnesses = []
        random_index = np.random.choice(np.arange(self.X.shape[0]),
                                        size=(BATCH_SIZE,))
        for character in self.population:
            result = character.forward(self.X[random_index])
            fitness = np.multiply(result, self.y[random_index])
            self.fitnesses.append(np.sum(np.max(fitness, axis=1)))
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
