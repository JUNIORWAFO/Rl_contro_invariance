from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class LinearUpdateEnv(gym.Env):
    def __init__(self, A, B, state_bounds, action_bounds,n_modes, alpha = 0.9, Init_step =10, max_steps_per_episode =100, rewards= [-0.01,-1, 1]):
        """
        Initializes the environment.

        Args:
            A: The state transition matrix.
            B: The control input matrix.
            target_state: The desired target state.
            state_bounds: A tuple containing the lower and upper bounds of the state.
            action_bounds: A tuple containing the lower and upper bounds of the action.
        """
        self.A = A
        self.B = B
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.max_steps_per_episode = max_steps_per_episode
        self.action_value = np.linspace(self.action_bounds[0], self.action_bounds[1], n_modes)
        self.action_space = gym.spaces.Discrete(n_modes)
        self.observation_space = gym.spaces.Box(low=self.state_bounds[0], high=self.state_bounds[1], shape=(self.A.shape[0],), dtype=np.float32)
        self.rewards = rewards
        self.alpha = 0.1
        self.reset_()
        self.Init_step = Init_step
    def reset_(self):
        """
        Resets the environment to a random initial state within the state bounds.

        Returns:
            The initial state.
        """
        self.state = np.random.uniform(low=self.state_bounds[0], high=self.state_bounds[1], size=self.observation_space.shape)
        self.history = self.state.reshape(-1, 2)
        self.traj = self.state.reshape(-1, 2)
        self.center = self.state
        self.alpha = 0.1
        self.Input = []
        self.poly = None
        self.counter = 0
        self.step = 0
        self.auto = 0
        self.Terminated = False
        self.Truncated = False
        return self.state, self.auto, self.center, self.Terminated, self.Truncated
    def compute_reward_2(self, next_state, boundary_violation):

        self.center = (self.alpha)*self.center + (1 - self.alpha) * self.state

        self.history=np.vstack([self.history,self.state.reshape(-1, 2)])

        val = self.history.shape[0]

        distance = np.linalg.norm(next_state - self.center)
        # Penalize boundary violations
        if not boundary_violation:
            self.auto = 2
            return self.rewards[1]/(1+np.exp(-(distance)**2))
        in_goal_region = is_point_inside_hull_2(self.history, next_state)
        # Reward reaching the goal region
        if in_goal_region:
            self.auto = 1
            return self.rewards[2]*(1+np.exp(-(distance)**2))
        return self.rewards[0]/(1+np.exp(-(distance)**2))

    def update(self, action):
        """
        Performs a step in the environment.

        Args:
            action: The action to take.

        Returns:
            A tuple containing the next state, reward, done flag, and additional info.
        """
        # Apply action constraints
        action = self.action_value[action]
        self.Input.append(action)
        # Update state based on the linear update map
        next_state = self.A @ self.state.reshape((2,1)) + self.B @ np.array(action).reshape((self.B.shape[1],1))
        self.traj=np.vstack([self.traj,self.state.reshape(1, 2)])
        boundary_violation =  self.observation_space.contains(list(next_state.reshape((2,))))
        # Compute reward
        reward = self.compute_reward_2(next_state, boundary_violation)
        self.state = next_state.reshape((2,))
        self.step += 1
        self.Terminated  = self.auto != 0
        self.Truncated  = self.auto == 0 and self.step == self.max_steps_per_episode
        done = self.Terminated or self.Truncated
        info = {}
        return self.state , reward, self.auto, self.center, self.Terminated, self.Truncated, done, info

class GridWorldEnv(gym.Env):

    def __init__(self, A, B, state_bounds, action_bounds,n_modes, alpha = 0.9, Init_step =10, max_steps_per_episode =100, rewards= [-0.01,-1, 1]):
        """
        Initializes the environment.

        Args:
            A: The state transition matrix.
            B: The control input matrix.
            target_state: The desired target state.
            state_bounds: A tuple containing the lower and upper bounds of the state.
            action_bounds: A tuple containing the lower and upper bounds of the action.
        """
        self.A = A
        self.B = B
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.max_steps_per_episode = max_steps_per_episode
        self.action_value = np.linspace(self.action_bounds[0], self.action_bounds[1], n_modes)
        self.action_space = gym.spaces.Discrete(n_modes)
        self.observation_space = gym.spaces.Box(low=self.state_bounds[0], high=self.state_bounds[1], shape=(self.A.shape[0],), dtype=np.float32)
        self.rewards = rewards
        self.alpha = 0.1
        self.reset()
        self.Init_step = Init_step

        self.render_mode = None

    def _get_obs(self):
        return {"state": self.state, "auto": self.auto, "center": self.center}

    def _get_info(self):
        return {}
    def compute_reward_2(self, next_state, boundary_violation):

        self.center = (self.alpha)*self.center + (1 - self.alpha) * self.state

        self.history=np.vstack([self.history,self.state.reshape(-1, 2)])

        val = self.history.shape[0]

        distance = np.linalg.norm(next_state - self.center)
        # Penalize boundary violations
        if not boundary_violation:
            self.auto = 2
            return self.rewards[1]/(1+np.exp(-(distance)**2))
        in_goal_region = is_point_inside_hull_2(self.history, next_state)
        # Reward reaching the goal region
        if in_goal_region:
            self.auto = 1
            return self.rewards[2]*(1+np.exp(-(distance)**2))
        return self.rewards[0]/(1+np.exp(-(distance)**2))
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.state = np.random.uniform(low=self.state_bounds[0], high=self.state_bounds[1], size=self.observation_space.shape)
        self.history = self.state.reshape(-1, 2)
        self.traj = self.state.reshape(-1, 2)
        self.center = self.state
        self.alpha = 0.1
        self.Input = []
        self.poly = None
        self.counter = 0
        self.step = 0
        self.auto = 0
        self.Terminated = False
        self.Truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        action = self.action_value[action]
        self.Input.append(action)
        # Update state based on the linear update map
        next_state = self.A @ self.state.reshape((2,1)) + self.B @ np.array(action).reshape((self.B.shape[1],1))
        self.traj=np.vstack([self.traj,self.state.reshape(1, 2)])
        boundary_violation =  self.observation_space.contains(list(next_state.reshape((2,))))
        # Compute reward
        reward = self.compute_reward_2(next_state, boundary_violation)
        self.state = next_state.reshape((2,))
        self.step += 1
        self.Terminated  = self.auto != 0
        self.Truncated  = self.auto == 0 and self.step == self.max_steps_per_episode
        done = self.Terminated or self.Truncated
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, self.Terminated, self.Truncated, info, done 

        
