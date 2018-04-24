import numpy as np
from cost_functions import trajectory_cost_fn
import time


class Controller():
    def __init__(self):
        pass

    # Get the appropriate action(s) for this state(s)
    def get_action(self, state):
        pass


class RandomController(Controller):
    def __init__(self, env):
        """ YOUR CODE HERE """
        self.env = env

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Your code should randomly sample an action uniformly from the action space """
        return self.env.action_space.sample()


class MPCcontroller(Controller):
    """ Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
    def __init__(self,
                 env,
                 dyn_model,
                 horizon=5,
                 cost_fn=None,
                 num_simulated_paths=10,
                 ):
        self.env = env
        self.dyn_model = dyn_model
        self.horizon = horizon
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths

    def get_action(self, state):
        """ YOUR CODE HERE """
        """ Note: be careful to batch your simulations through the model for speed """
        current_states = np.zeros((self.horizon, self.num_simulated_paths, self.env.observation_space.shape[0]))
        actions = np.zeros((self.horizon, self.num_simulated_paths, self.env.action_space.shape[0]))
        next_states = np.zeros((self.horizon, self.num_simulated_paths, self.env.observation_space.shape[0]))
        current_states[0, :, :] = state
        for h in range(self.horizon):
            # change for np.random.uniform
            actions[h, :, :] = np.random.uniform(self.env.action_space.low,
                                                 self.env.action_space.high,
                                                 (self.num_simulated_paths, self.env.action_space.shape[0]))

            next_states[h, :, :] = self.dyn_model.predict(current_states[h, :, :], actions[h, :, :])
            # change next current state with next states
            if h < self.horizon - 1:
                current_states[h + 1, :, :] = next_states[h, :, :]

        tc = trajectory_cost_fn(self.cost_fn, current_states, actions, next_states)
        min_cost = np.argmin(tc)
        
        return actions[0][min_cost]
