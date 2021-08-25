import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

Train_data = pd.read_csv('linear_data_train.csv').to_numpy()
Test_data = pd.read_csv('linear_data_test.csv').to_numpy()

X_train = Train_data[:, [0,1]]
Y_train = Train_data[:, 2]
X_test = Test_data[:, [0,1]]
Y_test = Test_data[:, 2]

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)


class Perceptron:
    def __init__(self):
        pass

    def train(self, itr, X_tarin, Y_tarin):
        x_range = np.arange(X_train[:, 0].min(), X_train[:, 0].max(), 0.1)
        y_range = np.arange(X_train[:, 1].min(), X_train[:, 1].max(), 0.1)
        x, y = np.meshgrid(x_range, y_range)

        self.W = np.random.rand(2)
        lr = 0.001
        MAE = []
        MSE = []

        for i in range(X_train.shape[0]):
            x_train = X_train[i].reshape(1, -1)
            y_pred = np.matmul(x_train, self.W)
            e = Y_train[i] - y_pred
            self.W = self.W + e * lr * X_train[i]

            Y_pred = np.matmul(X_train, self.W)
            mae = np.mean(np.square(Y_train - Y_pred))
            MAE.append(mae)

            mse = np.mean(np.abs(Y_train - Y_pred))
            MSE.append(mse)

            # plot
            ax.clear()
            z = x * model.W[0] + y * model.W[1]
            ax.plot_surface(x, y, z, alpha=0.8, rstride=1, cstride=1)
            ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c='m', marker='o')
            ax.set_xlabel('X0')
            ax.set_ylabel('X1')
            ax.set_zlabel('Y')
            plt.pause(0.001)

        plt.show()

        return MAE, MSE

    def predict(self, X_test):
        Y_pred = np.matmul(X_test, self.W)
        Y_pred[Y_pred > 0] = 1
        Y_pred[Y_pred < 0] = -1

        return Y_pred

    def evaluate(self, X_test, Y_test, metric):
        Y_pred = np.matmul(X_test, self.W)
        Y_pred = Y_pred.reshape(-1, 1)

        if metric == 'MAE':
            absolute_error = np.abs(Y_pred - Y_test)
            absolute_error = np.round(absolute_error, 1)
            evaluation = np.mean(absolute_error)

        if metric == 'MSE':
            squared_error = (Y_pred - Y_test) ** 2
            squared_error = np.round(squared_error, 1)
            evaluation = np.mean(squared_error)

        if metric == 'accuracy':
            Y_pred[Y_pred > 0] = 1
            Y_pred[Y_pred < 0] = -1
            evaluation = np.count_nonzero(Y_pred == Y_test) / len(Y_test)

        return evaluation

# 3D plot data
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(10, 70)

itration = 1000
model = Perceptron()
MAE, MSE = model.train(itration, X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = model.evaluate(X_test, Y_test, 'accuracy')
MSE_test = model.evaluate(X_test, Y_test, 'MSE')
MAE_test = model.evaluate(X_test, Y_test, 'MAE')
print("accuracy_test", accuracy)
print("MSE_test", MSE_test)
print("MAE_test", MAE_test)

plt.plot((np.arange(len(MAE))), MAE, c='b')
plt.grid(True)
plt.xticks(np.arange(0,len(MAE)+1,100))
plt.xlabel('iteration')
plt.ylabel('MAE')
plt.show()

plt.plot((np.arange(len(MSE))), MSE, c='b')
plt.grid(True)
plt.xticks(np.arange(0,len(MSE)+1,100))
plt.xlabel('iteration')
plt.ylabel('MSE')
plt.show()








