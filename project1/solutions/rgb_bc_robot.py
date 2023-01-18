from base import RobotPolicy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


class RGBBCRobot(RobotPolicy):
    """ Implement solution for Part3 below """

    def train(self, data):
        # Flatten the image
        X_train = data.get('obs')
        X_train = X_train.reshape(X_train.shape[0], -1)
        # Create PCA dimension reduction instance and process the data
        self.dimensionReduction_PCA = PCA(n_components=3, svd_solver='full')
        X_train = self.dimensionReduction_PCA.fit_transform(X_train)
        print(self.dimensionReduction_PCA.explained_variance_ratio_)
        # get the label
        y_train = data.get('actions').ravel()
        # Create LR instance
        self.policy = LogisticRegression(max_iter=10000)
        # train the model
        self.policy = self.policy.fit(X_train, y_train)

    def get_actions(self, observations):
        # preprocess the data
        X_test = self.dimensionReduction_PCA.transform(observations)
        # predict
        y_test = self.policy.predict(X_test)

        return y_test
