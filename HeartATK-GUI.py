import sys
import pickle
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QLineEdit, QMessageBox
from train import Gene

class HeartAttackGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Heart Attack Risk Prediction')
        self.setGeometry(100, 100, 700, 300)

        # Labels for input boxes
        self.metadata = self.loadMetadata('metadata.txt')
        self.labels = [' '.join(label.split()[0:-1]) for label in self.metadata]
        self.inputBoxes = {label: QLineEdit() for label in self.labels}

        # Button
        self.predictButton = QPushButton('Predict')
        self.predictButton.clicked.connect(self.predictHeartAttack)

        # Layout
        layout = QVBoxLayout()
        for label in self.labels:
            layout.addWidget(QLabel(label))
            layout.addWidget(self.inputBoxes[label])
        layout.addWidget(self.predictButton)

        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def loadMetadata(self, filename):
        # Read metadata from file
        with open(filename, 'r') as f:
            metadata = [line.strip() for line in f.readlines()]
        return metadata

    def predictHeartAttack(self):
        # Load trained model
        with open('best_gene.pickle', 'rb') as f:
            self.gene: Gene = pickle.load(f)

        # Get input values
        input_values = []
        for label in self.labels:
            value = float(self.inputBoxes[label].text())
            input_values.append(value)

        # Predict heart attack risk
        prediction = self.gene.forward(np.array(input_values).reshape(1, -1))

        # Display prediction result
        if np.argmax(prediction) == 1:
            QMessageBox.information(self, 'Prediction Result', 'You are at risk of heart attack.')
        else:
            QMessageBox.information(self, 'Prediction Result', 'You are not at risk of heart attack.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = HeartAttackGUI()
    ex.show()
    sys.exit(app.exec_())
