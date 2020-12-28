import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv('Resources/parkinsons.data')

# Get the features and labels
features = df.loc[:, df.columns != 'status'].values[:, 1:]
labels = df.loc[:, 'status'].values

# Get the count of each label (0 and 1) in labels
# print("Parkinson's (one): {}, healthy (zero): {}".format(labels[labels == 1].shape[0], labels[labels == 0].shape[0]))

# Scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)
