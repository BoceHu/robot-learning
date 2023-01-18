import abc


class RobotPolicy(abc.ABC):

    @abc.abstractmethod
    def train(self, data):
        """
            Abstract method for training a policy.

            Args:
                data: a dict that contains X (key = 'obs') and y (key = 'actions').

                X is either rgb image (N, 64, 64, 3) OR  agent & goal pos (N, 4)  

            Returns:
                This method does not return anything. It will just need to update the
                property of a RobotPolicy instance.
        """

    @abc.abstractmethod
    def get_action(self, obs):
        """
            Abstract method for getting action. You can do data preprocessing and feed
            forward of your trained model here.
            Args:
                obs: an observation (64 x 64 x 3) rgb image OR (4, ) positions
            
            Returns:
                action: an integer between 0 to 3
        """