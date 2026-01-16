# Digit-classifier via a very simple neural network 
Hey, I've used one hidden layer for training and I achieved an accuracy of 83%. I've used the MNIST dataset found on Kaggle. A better intuition and mathematical approach could be developed by watching 3Blue1Brown's video on neural networks and backpropagation. I wanted to build such a training system only through sheer code to understand (to some degree) what actually happens.

Below I've written a few notes that helped me understand things better:

# Dataset and Network Structure

-Every image in the MNIST database is a 28 x 28 pixel sized picture.

-So that's like a matrix, meaning we have a total of 784 different pixels with different colours — white and black.

-Each image's pixel is going to go into our input layer.

-The output layer will have 10 neurons (representing digits 0–9).

-The hidden layer also has 10 nodes.

-For the input-to-hidden layer, we use ReLU activation, and for the hidden-to-output layer we use softmax activation.

# Activation Functions

- Softmax activation converts vectors into probabilities, and that helps us in categorising (it calculates probabilities).

- It's only used as an output layer in a neural network.

- We consider the higher probability as the actual output.

# Environment and Setup

-I went to Jupyter Notebook next. I had to select a kernel (engine that actually runs the code), and I had two options that confused me:

-Pyodide: Python compiled to WebAssembly. Runs inside a web browser. WebAssembly is a low-level, portable binary format that runs fast and safely across platforms.

-XPython: An alternative Python interpreter (not CPython).

-I then switched to Google Colab — much easier!

# Libraries Used

-NumPy: Stores large numerical data efficiently, supports linear algebra, statistics, and random numbers, and improves performance.

-Pandas: Works with structured data (tables, CSV files). Loads data, stores them, cleans, filters, and analyses them.

-Seaborn: Used for statistical data visualisation. Creates attractive, informative plots. Built on top of Matplotlib.

-Pylab: An old convenience interface integrating NumPy and Matplotlib into a single namespace. No longer recommended. Instead of pylab, use explicit imports.

# Data Loading and Preprocessing

-I load all the training data using read_csv from the Pandas library.

-head() shows the first few rows of the dataset.

-data = np.array(data) converts the data into a NumPy array (raw values, no columns or indices), making matrix math easier.

-m, n = data.shape extracts the dimensions of the data array. It returns a tuple (rows, columns).

-np.random.shuffle(data) randomly reorders the elements of the array in-place.

-To control randomness (pseudo-randomness), we use np.random.seed().

# Data Splitting

-data_dev = data[0:1000].T   --> Selects the first 1000 rows from data and Transposes the matrix (Stores it in data_dev)

The data is split into development and training sets.

-After transposing, labels are separated into y, and the remaining rows are pixel values.

-Pixel values are normalised by dividing by 255 (scales them between 0–1) to improve speed.

Shapes and Conventions

-__, m_train = X_train.shape

- __ is a Python convention for a variable we don't care about.

# One-Hot Encoding

-The one_hot function converts a vector of class labels into one-hot encoded vectors.

-This is a way to represent a category using 0s and 1s.

-A better way to think about it: which class is this? without implying any numeric relationship.

-Y.astype(int) converts the data type of Y to integers.

-one_hot_Y = np.zeros((Y.size, Y.max() + 1)) --> Creates an empty matrix full of zeros.

-Y.size → number of examples

-Y.max() + 1 → number of classes

-one_hot_Y[np.arange(Y.size), Y] = 1  --> For each row, put a 1 in the column corresponding to its class.

-Why One-Hot Encoding? Neural networks do math. Class labels are categories. you need to make them numerical.

-One-hot encoding removes order and distance between classes.

# Backpropagation

-Backprop works backwards: Measure error at the output Push that error backward through the network Compute how much each weight contributed to the error

# other notes:

re is the regular expression module, used for pattern matching in strings.

re.A == re.ASCII (ASCII-only matching instead of full Unicode).

The update_params function is based on:

Moving parameters in the opposite direction of the gradient

The gradient points uphill (towards higher error)
