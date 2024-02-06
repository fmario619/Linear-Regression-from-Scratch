# write your code here
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


class CustomLinearRegression:

    def __init__(self, *,  fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.coefficient = None
        self.intercept = None

    def fit (self, X, y):
        X=np.array(X)
        y = np.array(y)


        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        X_T_X_inv = np.linalg.inv(X.T @ X)
        Beta = X_T_X_inv @ X.T @ y

        if self.fit_intercept:
            self.coefficient = Beta[1:]
            self.intercept = Beta[0]
        else:
            self.coefficient = Beta
            self.intercept = 0

    def predict(self, X):
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones if intercept is to be included
        return X @ np.r_[self.intercept, self.coefficient] if self.fit_intercept else X @ self.coefficient


    def r2_score(self, y, yhat):
        y = np.array(y)
        yhat = np.array(yhat)
        y_yhat_difference = y-yhat
        y_yhat_squared_difference = y_yhat_difference ** 2
        sum_y_yhat_squared_difference = np.sum(y_yhat_squared_difference)
        y_mean = np.mean(y)
        y_y_mean_difference = y - y_mean
        y_y_mean_difference_squared = y_y_mean_difference ** 2
        sum_y_y_mean_difference_squared = np.sum(y_y_mean_difference_squared)
        ratio = sum_y_yhat_squared_difference/sum_y_y_mean_difference_squared
        r2_calc = 1-ratio
        return r2_calc



    def rmse(self, y, yhat):
        y = np.array(y)
        yhat = np.array(yhat)
        difference = y - yhat
        sqrd_difference = difference ** 2
        mean_squared_diffrence = sqrd_difference.mean()
        rmse_calc = np.sqrt(mean_squared_diffrence)
        return rmse_calc




f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]

X_train =np.column_stack((f1, f2, f3))
y_train = y

# Creating an instance of CustomLinearRegression and fitting the data
model = CustomLinearRegression(fit_intercept=True)
model.fit(X_train, y_train)
model_intercept = model.intercept
model_coef = model.coefficient
y_pred = model.predict(X_train)
model_r2_score = model.r2_score(y_train, y_pred)
model_rmse = model.rmse(y_train, y_pred)

#Using SKlearn linear regression package
model_sk_learn = LinearRegression()
model_sk_learn.fit(X_train, y_train)
model_sk_learn_intercept = model_sk_learn.intercept_
model_sk_learn_coef = model_sk_learn.coef_
y_pred_sk = model_sk_learn.predict(X_train)
model_sk_learn_rmse = root_mean_squared_error(y_train, y_pred_sk)
model_sk_learn_r2 = r2_score(y_train, y_pred_sk)

# Printing the model parameters differences
dict = {'Intercept': model_sk_learn_intercept - model_intercept, 'Coefficient': model_sk_learn_coef -model_coef, 'R2': model_sk_learn_r2 - model_r2_score, 'RMSE': model_sk_learn_rmse-model_rmse}
print(dict)