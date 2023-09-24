import keras
import visualkeras

from numpy import loadtxt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten


class DataSet:
    'Dataset class, contains numpy dataset and reformatting functions'
    def init(self, source : str):
        self.source = source
        self.data = loadtxt(self.source, delimiter=',')


class NeuralNet:
    'The main neural net class, contains building and prediction functions'

    def __init__(self, directory: str, loadQ=None):
        'Directory = directory to the csv file, loadQ = keras model to boot from, requires directory to run eval functions'
        self.model = None
        self.directory: str = directory
        self.bounds: int = len(loadtxt(directory, delimiter=",")[0]) - 1
        self.dataset = DataSet(directory)

        if loadQ:
            self.model = keras.models.load_model(loadQ)

    def build_base(self, d1=12, d2=8, d3=1, a1="relu", a2="relu", a3="sigmoid"):
        'Defines a basic sequential keras model in a [[0,12], [0,8], [0,1]] shape'
        model = Sequential()
        model.add(Dense(d1, input_shape=(self.bounds,), activation=a1))
        model.add(Dense(d2, activation=a2))
        model.add(Dense(d3, activation=a3))

        self.model = model

    def compile(self):
        'Compiles the model for training and/or prediction'
        self.model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    #

    def train(self, epochs: int=150, batch_size: int =10, verbose: int=0):
        'Fits the created model to the dataset'
        dataset = self.dataset.data

        # splits the dataset into input (X) and output (y) datasets based of the length of the csv file
        X = dataset[:, 0:self.bounds]
        y = dataset[:, self.bounds]

        # fits the keras model on the datasets
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Custom construction functions

    def init_model(self):
        'Defines model as Sequential'
        model = Sequential()
        self.model = model

    def add_dense(self, dim: int=12, act: str="relu"):
        self.model.add(Dense(dim, input_shape=(self.bounds,), activation=act))

    # Model save and load functions

    def save(self, name: str="model.keras"):
        'Saves keras model'
        self.model.save(name)

    def load(self, name: str="model.keras"):
        'Loads model from directory'
        self.model = keras.models.load_model(name)

    # Eval and prediction functions

    def eval(self):
        'Evaluates the models accuracy and loss, returns a tuple'
        dataset = self.dataset.data

        # splits dataset into input (X) and output (y) datasets
        X = dataset[:, 0:self.bounds]
        y = dataset[:, self.bounds]

        # evaluate the keras model
        loss, accuracy = self.model.evaluate(X, y, verbose=0)

        return loss, accuracy

    def predict_csv(self, input_directory: str):
        'Predicts responses from an inputted csv file'

        #  an additional "blank" row filled with zeros may be needed if the csv is 1 row long (edge case)
        x = loadtxt(input_directory, delimiter=",")
        predictions = (self.model.predict(x, verbose=0) > 0.5).astype(int)

        return predictions

    def predict_array(self, array: list):
        'Predicts responses from an array'

        # converting all array items to "str" type for dataset writing
        array = [str(x) for x in array]

        # assembly of dataset
        open('neural_net.csv', 'w').close()
        with open("neural_net.csv", "a") as f:
            f.write(",".join(array) + "\n")
            f.write(("0,"*(len(array)-1)) + "0")
        x = loadtxt("neural_net.csv", delimiter=",")

        predictions = (self.model.predict(x, verbose=0) > 0.5).astype(int)
        return predictions[:-1]

    def visualise(self):
        'Visualises the keras model'
        model=self.model

        visualkeras.layered_view(model).show()

    def summarise(self):
        'Returns a summary of the model'
        self.model.summary()

    def kill(self):
        'Deletes the model object, does NOT save the model in a directory'
        self.model = None

    def ensure_future_main(self, name="future.keras", verbose=0):
        'Deletes the model, but saves it in a directory'

        if verbose:
          self.model.summarise()

        self.model.save(name)
        self.model = None

