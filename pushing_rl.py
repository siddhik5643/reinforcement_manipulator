import numpy as np
from tqdm import tqdm

from gym.spaces import Box
# PPO Implementation reference:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# DQN Implementation reference:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


from panda_pushing_env import TARGET_POSE, OBSTACLE_CENTRE, OBSTACLE_HALFDIMS, BOX_SIZE


class RandomPolicy(object):
    """
    A random policy for any environment.
    It has the same method as a stable-baselines3 policy object for compatibility.
    """

    def __init__(self, env):
        self.env = env

    def predict(self, state):
        action = self.env.action_space.sample()  # random sample the env action space
        return action, None

def execute_policy(env, policy, num_steps=20):
    states = []
    rewards = []
    goal_reached = False
    
    # Reset the environment to get the initial state
    state= env.reset()
    
    for _ in tqdm(range(num_steps)):
        # Get action from policy
        action, _ = policy.predict(state)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        
        # Store state and reward
        states.append(state)
        rewards.append(reward)
        
        # Check if goal is reached
        if done:
            goal_reached = True
            break
        
        # Update state
        state = next_state
    
    return states, rewards, goal_reached

def obstacle_free_pushing_reward_function_object_pose_space(state, action):
    object_pos = state[:2]
    target_pos = TARGET_POSE[:2]
    reward = 0;
    distance_to_target = np.linalg.norm(object_pos - target_pos)
    
    # Reward inversely proportional to the squared distance (smooth gradient)
    reward = -30 * (distance_to_target ** 2)
    
    # Bonus for making progress (difference in distance before and after action)
    previous_distance = state[-1]  # Assuming the last state element stores the previous distance
    progress = previous_distance - distance_to_target
    reward += 40 * progress  # Reward for moving closer
    #print(f'Distance {progress}')
    # Large bonus for reaching the target
    if distance_to_target < BOX_SIZE:
        reward += 150
    
    # Small penalty for excessive movement (encourages efficiency)
    #movement_penalty = np.linalg.norm(action) * 0.1
    #reward -= movement_penalty
    #print(f'distance {state[-1]}')
    #print(f'Distance {progress}')
    return reward





def pushing_with_obstacles_reward_function_object_pose_space(state, action):
    """
    Defines the state reward function for the action transition (prev_state, action, state)
    :param state: numpy array of shape (state_dim)
    :param action:numpy array of shape (action_dim)
    :return: reward value. <float>
    """
    reward = None
    # --- Your code here



    # ---
    return reward

# Ancillary functions
# --- Your code here



# ---