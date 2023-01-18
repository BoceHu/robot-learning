from base import Regressor
import numpy as np
from sklearn.linear_model import LinearRegression


class PositionRegressor(Regressor):
    """ Implement solution for Part 1 below  """

    def train(self, data):
        """
        data: dictionary
        key: obs; actions; info

        value:
        obs.shape:(500, 64, 64, 3)
        action.shape:(500,)
        info value example:{'agent_pos': (0.7100000381469727, -0.4000000059604645),
        'obstacle_infos': [{'size': array([0.1, 0.1]), 'x': 0.3, 'y': -0.1},
        {'size': array([0.1, 0.1]), 'x': 0.7, 'y': 0.1}],
        'goal': array([0.2, 0.4])}
        """
        # get input data
        X_train = data.get('obs')
        # preprocess the image (flatten) from (size,64,64,3) -> (size,-1)
        X_train = X_train.reshape(X_train.shape[0], -1)
        # get label data list
        y_train = []
        for info in data.get('info'):
            y = info.get('agent_pos')
            y_train.append(y)

        # create LinearRegression instance
        self.regressor = LinearRegression()
        # train the model
        self.regressor = self.regressor.fit(X_train, y_train)

    def predict(self, Xs):
        # preprocess the input image
        X_test = Xs.reshape((Xs.shape[0], -1))
        # predict
        y_test = self.regressor.predict(X_test)
        return y_test
