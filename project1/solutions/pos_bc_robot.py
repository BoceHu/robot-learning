from base import RobotPolicy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


class POSBCRobot(RobotPolicy):
    """ Implement solution for Part 2 below """

    def train(self, data):
        """
        data key: obs; actions
        obs example:[ 0.11000001 -0.30000001]
        action example: [1]
        """
        # get the label and ravel the label list. (500,1) -> (500,)
        y_train = data.get('actions').ravel()
        # get the input data
        X_train = data.get('obs')
        # create PolynomialFeatures' instance
        self.poly = PolynomialFeatures(3)
        # create LR instance
        self.policy = LogisticRegression()
        # self.policy=SGDClassifier(max_iter=2000)
        # complicate the input data (add more features)
        X_train = self.poly.fit_transform(X_train)
        # train the model
        self.policy = self.policy.fit(X_train, y_train)

    def get_actions(self, observations):
        # preprocess the data
        X_test = self.poly.fit_transform(observations)
        # predict
        y_test = self.policy.predict(X_test)

        return y_test
