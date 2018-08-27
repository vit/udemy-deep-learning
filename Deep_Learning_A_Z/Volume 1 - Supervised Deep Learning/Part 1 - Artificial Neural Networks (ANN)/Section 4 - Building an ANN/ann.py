

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Part 1 -- Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable

# replace country an gender with integer numbers
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# replace Country with 3 binary columns isFrench/isGerman/isSpanish
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# remove 1 redundant column
X = X[:,1:]



# Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 -- Make the ANN

# import keras libraries
import keras
# Sequential -- to initialize the network
from keras.models import Sequential
# Dense -- to build layers
from keras.layers import Dense


# initializing the ANN
# there are two ways to initialize layers
# -- define sequence of layers
# -- define a graph

classifier = Sequential()


# adding the input layer and the first hidden layer
# rectifier function for hidden layers
# sigmoid function for output layer (gives probability)

# no rules for optimal number of inputs in the hidden layer, it's the art

# if your data is lineary separable,
# you don't need hidden layer and neural network

# tip : choose average of the number of nodes of input layer
# and the number of nodes of output layer
#
# or use parameters tuning -- take cross-validation set
# and test different models with different model parameters

classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
# UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform")`


# adding the second hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
# UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", units=6, kernel_initializer="uniform")`

# second hidden layer is not nececcerily useful;
# we add it for two reasons:
# -- it's deep learning
# -- to know how to do it if we need it


# adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
# UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="sigmoid", units=1, kernel_initializer="uniform")`

# if the depending varible has more than 2 categories (3 for example),
# output_dim=3 -- number of classes,
# activation='softmax' -- sigmoid function too, but for more than 2 categories


# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# adam -- one of gradient descent algorithms
# binary_crossentropy - logarithnic loss function
# (if more than two categories -- categorical_crossentropy)


# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)







# Part 3 -- Making the predictions and evaluating the model


# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# we get accuracy 84--86% on test set


