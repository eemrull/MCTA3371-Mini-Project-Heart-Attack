import pickle
from utils import *
from train import Character

with open('best_character.pickle', 'rb') as f:
    character: Character = pickle.load(f)

# if __name__ == '__main__':
    X_test, X_train, y_test, y_train = load_data()

    # character = Character()
    correct = 0
    for X, y in zip(X_test, y_test):
        value = character.forward(X)
        if np.argmax(value.flatten()) == np.argmax(y):
            correct += 1
        print(f'{value} {np.argmax(value.flatten())} {y}')
    print(correct/len(y_test))
    # print(character.nn[0].weights)