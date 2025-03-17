#Initiation of the KNN Class, using the OOP paradigm specified in the assignment description

class KNN:

    # Constructor
    def __init__(self, K, distance_func):
        self.K = K
        self.distance_func = distance_func
        return

    # Defining the 'fit' function - trains the model by merely remembering the dataset.
    def fit(self, x, y):
        self.x = x
        self.y = y
        #self.C = y.max() + 1 # C = the number of classes
        self.C = int(self.y.max()) + 1
        return self

    # Defining the 'predict' function
    def predict(self, x_test): #pass in either x_penguin_test or x_heart_test
        """
        Makes a prediction using the stored training data and the test data given as an argument.
        """
        num_test = x_test.shape[0]

        distances = self.distance_func(self.x.values[None, :, :],
                                 x_test.values[:, None, :])


        knns = pd.DataFrame(0, index=range(num_test), columns=range(self.K), dtype=int)


        y_prob = pd.DataFrame(index=range(num_test), columns=range(self.C), dtype=float).fillna(0)

        for i in range(num_test):

            knns.iloc[i, :] = distances[i].argsort()[:self.K]


            neighbor_labels = self.y.iloc[knns.iloc[i, :]].values
            class_counts = pd.Series(neighbor_labels).value_counts().reindex(range(self.C), fill_value=0)

            y_prob.iloc[i, :] = class_counts / self.K

        return y_prob, knns

    # accuracy evaluation
    def evaluate_acc(self, y_true, y_pred):
        accuracy = (y_pred == y_true).mean()
        print(f'accuracy is {accuracy * 100:.1f}.')
        return accuracy
