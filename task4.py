import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_data():
    """Generates data points for the function y = x^2 + noise
    returns x_data(200,1) and y_data(200,1))"""
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] #From task description
    noise = np.random.normal(0, 0.02, x_data.shape) #From task description
    y_data = np.square(x_data) + noise #From task description
    return x_data, y_data #Return x_data and y_data

def linear_regression(x_train, y_train, x_test, y_test):
    """Linear regression with mean squared error
    args(x_train(160,1), y_train(160,1), x_test(40,1), y_test(40,1))
    returns y_pred_lin(40,1), mse_lin(1,1))"""
    lin_reg = LinearRegression() #SK learn linear regression model
    lin_reg.fit(x_train, y_train) #Fit the model to the training data
    y_pred_lin = lin_reg.predict(x_test) #Predict y values for the test data
    mse_lin = mean_squared_error(y_test, y_pred_lin) # Calculate mean squared error
    return y_pred_lin, mse_lin # Return predictions and mean squared error

def polynomial_regression(x_train, y_train, x_test, y_test):
    """Polynomial regression with degree 2 and mean squared error
    args(x_train(160,1), y_train(160,1), x_test(40,1), y_test(40,1))
    returns y_pred_poly(40,1), mse_poly(1,1))"""
    poly_features = PolynomialFeatures(degree=2) #sklearn polynomial features with degree 2 <- from task description
    x_poly_train = poly_features.fit_transform(x_train) #Transform x_train to polynomial features
    x_poly_test = poly_features.transform(x_test) #Transform x_test to polynomial features
    poly_reg = LinearRegression() #SK learn linear regression model
    poly_reg.fit(x_poly_train, y_train) #Fit the model to the training data
    y_pred_poly = poly_reg.predict(x_poly_test) #Predict y values for the test data
    mse_poly = mean_squared_error(y_test, y_pred_poly) # Calculate mean squared error
    return y_pred_poly, mse_poly # Return predictions and mean squared error

def neural_network(x_train, y_train, x_test, y_test):
    """Neural network with 1 hidden layer with 6 nodes and mean squared error
    args(x_train(160,1), y_train(160,1), x_test(40,1), y_test(40,1))
    returns y_pred_nn(40,1), mse_nn(1,1))"""
    model = Sequential()
    model.add(Dense(6, input_dim=1, activation='relu'))  # 1 hidden layer with 6 nodes
    model.add(Dense(1, activation='linear'))  # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=100, batch_size=10)
    y_pred_nn = model.predict(x_test)
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    return y_pred_nn, mse_nn

def plot_results(x_test, y_test, y_pred_lin, y_pred_poly):
    """Plots data points and curve of predictions for Linear and Polynomial Regression
    args(x_test(40,1), y_test(40,1), y_pred_lin(40,1), y_pred_poly(40,1))"""
    plt.scatter(x_test, y_test, color='blue', label='Actual Data')
    plt.plot(x_test, y_pred_lin, color='red', label='Linear Regression')
    plt.plot(x_test, y_pred_poly, color='green', label='Polynomial Regression')
    plt.title('Linear and Polynomial Regression')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    # (0) Generate data points in Python using:
    # x_data = np.linspace(−0.5, 0.5, 200)[:, np.newaxis]
    # noise = np.random.normal(0, 0.02, x_data.shape)
    # y_data = np.square(x_data) + noise
    # From the task description this is the data we are supposed to use go to generate_data() function to see the code
    x_data, y_data = generate_data()
    # Split the dataset into training (80%) and testing (20%) data
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0)
    
    # (a) Linear Regression, Establish a linear regression model to predict y data (you can use sklearn).
    y_pred_lin, mse_lin = linear_regression(x_train, y_train, x_test, y_test)   
    # From the task description we are supposed to use sklearn for linear regression, to predict y data and calculate mean squared error

    # (b) Polynomial Regression with degree 2 use sklearn to predict y data and calculate mean squared error
    y_pred_poly, mse_poly = polynomial_regression(x_train, y_train, x_test, y_test)

    # (c) Neural Network with 1 hidden layer with 6 nodes use keras to predict y data and calculate mean squared error
    # Spiltting the dataset into training 80/20 data we did earlier

    y_pred_nn, mse_nn = neural_network(x_train, y_train, x_test, y_test)
    # (d) Calculate and compare mean squared errors
    # Calculate the mean squared error for each model we did earlier in each model function and return it
    # Now we can compare the mean squared errors for each model in print statements down below
    print("Mean Squared Error (Linear Regression):", mse_lin)
    print("Mean Squared Error (Polynomial Regression):", mse_poly)
    print("Mean Squared Error (Neural Network):", mse_nn)

    # (e) Plot data points and curve of predictions for Linear and Polynomial Regression
    plot_results(x_test, y_test, y_pred_lin, y_pred_poly)