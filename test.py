import pickle
from utils import *
from train import Gene

with open('best_gene.pickle', 'rb') as f:
    gene: Gene = pickle.load(f)

# if __name__ == '__main__':
    X_test, X_train, y_test, y_train = load_data()

    # gene = Gene()
    correct = 0
    for X, y in zip(X_test, y_test):
        value = gene.forward(X)
        if np.argmax(value.flatten()) == np.argmax(y):
            correct += 1
        print(f'{np.argmax(value.flatten())} {np.argmax(y)} {np.max(value)} {value} {y}')
    print(correct/len(y_test))
    print(len(y_train))
    print(len(y_test))
    # print(gene.nn[0].weights)