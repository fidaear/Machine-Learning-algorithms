import numpy as np
class LogisticRegression():

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def __init__(self,lr=0.001,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self,X,y):
        n_samples,n_features=X.shape
        self.weights=np.zeros(self.features)
        self.bias= 0

        for _ in range(self.n_iters):
            linear_prd=np.dot(X, self.weights)+self.bias
            predictions=sigmoid(linear_prd)


    def predict():


    