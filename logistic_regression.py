import math
import numpy as np
from typing import List
# importing toy dataset iris
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# add in the sigmoid implementation
def sigmoid(x, theta):
    """
    Implements the sigmoid function

    h(x[i], theta) = 1 / (1 + e^-theta_transpose * x)

    :param x: numpy array of input vector.
    :param theta: numpy array of model weights vector.

    :return: the probability value between 0 and 1 based on the sigmoid function graph.
    :rtype: float
    """
    theta_x = -1 * (theta.transpose().dot(x))
    return 1 / (1 + (math.e ** theta_x))


# add in the implementation of cost function
def cost_function(x, y, theta):
    """
    Implements the cost function for the Logistic Regression model.

    J(theta) = -1/m(y(i)*log(h(x(i), theta)) + (1- y(i))*log(h(x(i), theta))
    where m = number of samples
    h = sigmoid function
    theta = weight vector

    For every value of input X, label Y and the weight theta, a cost will be calculated
    which will further be used by the optimizer/solver algorith to figure out the best possible value of
    theta (because that's the only thing variable here).

    :param x: numpy array of input vector.
    :param y: numpy array of output vector.
    :param theta: numpy array of model weights vector.

    :return:the cost value J(theta)
    :rtype: float
    """
    m = y.shape[0]
    cost_sum = 0
    for i in range(0, m):
        # print('m ', sigmoid(x[i], theta))
        cost_sum += y[i] * math.log(sigmoid(x[i], theta)) + (1 - y[i]) * math.log(1 - sigmoid(x[i], theta), 10)
    return -1 * (1 / m) * cost_sum


# add in the implementation of cost function derivative
def cost_derivative(x, y, theta, j):
    """
    Implements the derivative of the LR cost function.

    :param j: the position of weight value at index i
    :type j: int
    :param x: input feature vector.
    :type x: np.array
    :param y: input feature labels.
    :type y: np.array
    :param theta: model weight vector.
    :type theta: np.array
    :return: the value of derivative basis the values of theta, x and y.
    :rtype: float
    """
    m = y.shape[0]
    der_sum = 0
    for i in range(0, m):
        der_sum += (sigmoid(x[i], theta) - y[i]) * x[i][j]

    return (1 / m) * der_sum


# add in the implementation of gradient decent
def gradient_descent(x, y, theta, alpha, n_epochs):
    """
    Implements the stochastic gradient descent algorithm.

    gradient descent is defined by the following equation.
    for every possible value of j
    theta_j = theta_j - (alpha/m) * summation(1 -> m) (h(x_i, theta) - y_i)*x_ij

    :param n_epochs: The number of epochs used by the optimizer to get at the local minimum
    :type n_epochs: int
    :param x: input feature vector.
    :type x: np.array
    :param y: input feature labels.
    :type y: np.array
    :param theta: model weight vector.
    :type theta: np.array
    :param alpha: learning rate i.e, how fast the equation converges.
    :type alpha: float

    :return: the local minimum value for theta for basis the cost function minimization.
    :rtype: float
    """
    m = x.shape[1]
    new_theta = []
    op_theta = np.zeros(m)
    for j in range(m):
        print(f"Optimizing for theta/weight at index {j}")
        for epoch in range(n_epochs):
            theta[j] = theta[j] - (alpha * cost_derivative(x, y, theta, j))
            loss = cost_function(x, y, theta)
            print(f"For epoch number {epoch}: loss = {loss}, theta = {theta[j]}")

    return theta


class CustomLogisticRegression:
    """
    solver: stochastic gradient decent.
    prediction function: sigmoid
    """

    def __init__(self, x, y, n_epochs):
        self.x = x
        self.y = y
        self.n_epochs = n_epochs
        self.theta = np.zeros(x.shape[1])

    def fit(self):
        """
        This function is used to fit the input data for prediction of output Y.

        X*w + gradient = Y^hat
        where w = sigmoid function i.e, h(x[i], theta) = 1 / (1 + e^-theta_transpose * x)

        Based on the y^hat calc we compute the loss function/cost function:-
        J(theta) = -1/m(y(i)*log(h(x(i), theta)) + (1- y(i))*log(h(x(i), theta))
        where m = number of samples

        the optional level of theta is derived using the stochastic gradient decent
        algorithm, (SGD)

        h = h(X, theta)
        gradient = 1/m * X_transpose * (h - y)
        theta = theta - alpha * gradient where alpha = learning rate
        J(theta)

        :return:
        :rtype:
        """
        self.theta = gradient_descent(self.x, self.y, self.theta, alpha=0.01, n_epochs=self.n_epochs)
        print(f'The weight values after training are = {self.theta}')

    def transform(self, input_x):
        output_y = []
        for i in range(0, input_x.shape[0]):
            output_y.append(sigmoid(input_x[i], self.theta))
        return output_y

    def get_class_prediction(self, input_x, threshold=0.5):
        output_y = []
        for i in range(0, input_x.shape[0]):
            output_y.append(sigmoid(input_x[i], self.theta))
        return [1 if x > threshold else 0 for x in output_y]


# Driver function to train a custom LR on Iris dataset
data, target = load_breast_cancer(return_X_y=True, as_frame=True)
print(data.head())
print(target)

# splitting the data into train and test samples
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
print(f"shape of X_train = {X_train.shape} and shape of y_train = {y_train.shape}")
print(f"shape of X_test = {X_test.shape} and shape of y_test = {y_test.shape}")

# scaling the input features
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# since the logistic regression we are training have the y-intercept as 0 so
# the output that we need as a result of training the model is the values of theta or the feature weights.
n_epochs_input = 100
lr = CustomLogisticRegression(X_train, y_train.to_numpy(), n_epochs_input)
lr.fit()
y_hat = lr.get_class_prediction(X_test)

score_count = 0
for i in range(len(y_hat)):
    if y_hat[i] == y_test.to_list()[i]:
        score_count += 1
print(f"accuracy of prediction is {score_count/len(y_test)}")


