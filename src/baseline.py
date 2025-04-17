import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import os
import logging
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

from demo import load_demos
from Grasper.wrappers import BetterExploration, HandParams, TaskType


BASELINE_NAME = "baseline_model"
VIDEOS_FOLDER = "./videos"


class BaselineModel(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, sum(output_dims))
        self.output_dims = output_dims

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        segments = torch.split(x, self.output_dims, dim=1)
        softmax_segments = [torch.softmax(segment, dim=1) for segment in segments]
        x = torch.cat(softmax_segments, dim=1)
        return x
    
    # Used for evaluation using stable-baselines3 API
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        if isinstance(observation, np.ndarray):  # Convert to tensor if needed
            observation = torch.tensor(observation, dtype=torch.float32)

        with torch.no_grad():  # No gradient computation
            action_probs = self.forward(observation)

        if deterministic:
            segments = torch.split(action_probs, self.output_dims, dim=1)
            argmax_segments = [torch.argmax(segment, dim=1) for segment in segments]
            action = torch.cat(argmax_segments, dim=-1)[None, :]
        else:
            segments = torch.split(action_probs, self.output_dims, dim=1)
            argmax_segments = [torch.distributions.Categorical(segment).sample() for segment in segments]
            action = torch.cat(argmax_segments, dim=-1)[None, :]

        return action.numpy(), None


class TrajectoryDataset(Dataset):
    def __init__(self, states, actions):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def _make_env(env_id, hand_type, task_type, record=False):
    if not record:
        env = gym.make(env_id)
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env = BetterExploration(env)
    env = HandParams(env, hand_type)
    env = TaskType(env, task_type)
    env = gym.wrappers.FlattenObservation(env)
    if record:
        env = gym.wrappers.RecordVideo(env, video_folder=VIDEOS_FOLDER, episode_trigger=lambda x: True, disable_logger=True)
    return env


def load_baseline(obs_space, action_space, device, verbose=True):
    # Load the baseline model
    model_path = f"{BASELINE_NAME}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("No saved model or checkpoints found.")

    # Load model
    if verbose:
        print(f"Loading baseline from {model_path}")
    model = BaselineModel(obs_space, action_space)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    return model


def train_baseline(env_id, demo_index, epochs=2000):

    # Create the environment to extract the observation and action space
    env = gym.make(env_id)
    observation_space = env.unwrapped.observation_space.shape[0]
    action_types = env.unwrapped.action_space.shape[0]
    action_spaces = env.unwrapped.action_space.nvec

    # Create the demo dataset
    seeds, demo_matrix = load_demos(env_id, demo_index)
    states = demo_matrix[:, :observation_space, :].transpose(0, 2, 1).reshape(-1, observation_space)
    action_labels = demo_matrix[:, observation_space:, :].transpose(0,2,1).reshape(-1, action_types).astype(np.int32)
    # Remove "do nothing" actions
    do_nothing_mask = ~np.all(action_labels==0, axis=1)
    states = states[do_nothing_mask]
    action_labels = action_labels[do_nothing_mask]
    # Convert actions to one-hot encoding
    actions = np.zeros((action_labels.shape[0], action_spaces.sum()), dtype=np.float32)
    action_offset = 0
    for action in range(action_types):
        action_space = action_spaces[action]
        actions[:, action_offset:action_offset+action_space] = np.eye(action_space)[action_labels[:, action]]
        action_offset += action_space
    dataset = TrajectoryDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Train the baseline model to emulate the expert
    policy = BaselineModel(observation_space, action_spaces.tolist())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    policy.train()

    # Train on each demo for a number of epochs
    for epoch in range(epochs):
        num_wrong = 0
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            predicted_actions = policy(batch_states)
            loss = criterion(predicted_actions, batch_actions)
            num_wrong += -(torch.round(predicted_actions) - torch.round(batch_actions)).sum().item()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Num Wrong: {num_wrong}")

    # Save the model
    torch.save(policy.state_dict(), f"{BASELINE_NAME}.pth")
    print(f"Baseline model saved to {BASELINE_NAME}.pth")


def test_baseline(env_id, hand_type, task_type, n_eval_episodes=25, n_displayed_episodes=5):
    # Create the environment to extract the observation and action space
    env = gym.make(env_id)
    observation_space = env.unwrapped.observation_space.shape[0]
    action_types = env.unwrapped.action_space.shape[0]
    action_spaces = env.unwrapped.action_space.nvec.tolist()

    # Load the baseline model
    model = load_baseline(observation_space, action_spaces)

    # Evaluate model
    env = make_vec_env(lambda: _make_env(env_id, hand_type, task_type), n_envs=1, vec_env_cls=DummyVecEnv)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Render the environment
    gym.logger.min_level = logging.ERROR
    env = _make_env(env_id, hand_type, task_type, record=True)
    episodes = 1
    obs, info = env.reset()
    step = 0
    while True:
        action, _states = model.predict(obs[None, :], deterministic=False)
        obs, reward, done, terminated, info = env.step(action[0])
        step += 1
        env.render()
        if done or terminated:
            if episodes >= n_displayed_episodes:
                break
            obs, info = env.reset()
            episodes += 1
    env.close()
    print(f"Video saved to {VIDEOS_FOLDER} folder.")