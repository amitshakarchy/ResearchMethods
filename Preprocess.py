import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class PreProcessing:

    def __init__(self):
        # Read the data
        self.df = pd.read_csv('Resources/parkinsons.data')

    def pre_process(self):
        # Get the features and labels
        self.features = self.df.loc[:, self.df.columns != 'status'].values[:, 1:]
        self.labels = self.df.loc[:, 'status'].values

        # Get the count of each label (0 and 1) in labels
        # print("Parkinson's (one): {}, healthy (zero): {}".format(labels[labels == 1].shape[0], labels[labels == 0].shape[0]))

        # Scale the features to between -1 and 1
        self.scaler = MinMaxScaler((-1, 1))
        self.x = self.scaler.fit_transform(self.features)
        self.y = self.labels
        # Split the dataset
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=7)
