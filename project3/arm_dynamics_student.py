from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from models import build_model


class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = build_model(num_links, time_step)
        # ---
        self.model_loaded = True
        best_check_point = torch.load(model_path)
        self.model.load_state_dict(best_check_point)

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            self.model.eval()
            state = np.array(state).reshape(1, -1)
            action = np.array(action).reshape(1, -1)
            X_predict = np.concatenate((state, action), axis=1)
            # print(X_predict.shape)
            new_state = self.model(torch.from_numpy(X_predict).float()).detach().numpy()

            return new_state.reshape(-1, 1)

        else:
            return state
