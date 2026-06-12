import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IDSEnv(gym.Env):

    def __init__(self, X, y):
        super().__init__()

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)

        self.num_samples  = len(X)
        self.num_features = X.shape[1]
        self.current_index = 0

        self.visit_order = np.arange(self.num_samples)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_features,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.visit_order = self.np_random.permutation(self.num_samples)
        self.current_index = 0

        first_observation = self.X[self.visit_order[0]]
        return first_observation, {}

    def step(self, action):

        real_label = self.y[self.visit_order[self.current_index]]

        reward = self._compute_reward(int(action), int(real_label))

        self.current_index += 1

        episode_finished = (self.current_index >= self.num_samples)

        if episode_finished:
            next_observation = np.zeros(self.num_features, dtype=np.float32)
        else:
            next_observation = self.X[self.visit_order[self.current_index]]

        info = {
            'etiqueta_real': int(real_label),
            'accion_tomada': int(action)
        }

        return next_observation, reward, episode_finished, False, info

    def _compute_reward(self, action, real_label):

        if action == 1 and real_label == 1:
            return 1.0

        elif action == 0 and real_label == 0:
            return 0.1

        elif action == 1 and real_label == 0:
            return -0.5

        else:
            return -1.0

    def render(self):
        pass

    def close(self):
        pass
