import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Basic Tkinter GUI for file selection
def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Text files", "*.txt")])
    return file_path

# Load data from a file selected through Tkinter
def load_data():
    file_path = open_file_dialog()
    data = np.loadtxt(file_path)
    return data

# Visualize data using seaborn
def visualize_data(data):
    sns.scatterplot(x=data[:, 0], y=data[:, 1])
    plt.title("Data Visualization")
    plt.show()

# Use Plotly for interactive plots
def interactive_plot(data):
    fig = px.scatter(x=data[:, 0], y=data[:, 1], title="Interactive Plot")
    fig.show()

# Linear Regression using scikit-learn
def linear_regression(data):
    X, y = data[:, 0].reshape(-1, 1), data[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Visualize the regression line
    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # Print mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    # Load data
    data = load_data()

    # Visualize data
    visualize_data(data)

    # Interactive plot
    interactive_plot(data)

    # Linear Regression
    linear_regression(data)
