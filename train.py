import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3 import PPO
import numpy as np

# Custom Wrapper to handle environment compatibility and apply Frame Skip
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        
    def reset(self, **kwargs):
        # Standard environment reset
        obs = self.env.reset()
        
        # Handle Gym backwards compatibility (tuple unwrapping)
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
            
        return obs, info
        
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # Repeat the same action 'skip' times to accelerate learning (Frame Skipping)
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        # Reward Shaping:
        # Apply a small penalty if the agent stands still or gets stuck
        if total_reward <= 0:
            total_reward -= 0.1
            
        # Return the standard 5 values expected by the Gym API for SB3
        return obs, total_reward, done, False, info

class DistanceTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_distances = []

    def _on_step(self) -> bool:
        # Check the info dict from the vectorized environments at each step
        for info in self.locals.get("infos", []):
            if "x_pos" in info:
                # If the agent dies or finishes the level, record the final x_pos
                if "terminal_observation" in info:
                    self.episode_distances.append(info["x_pos"])
        return True

    def _on_rollout_end(self) -> None:
        # Triggered every time PPO updates its policy (i.e., after rollout collection)
        if len(self.episode_distances) > 0:
            average = np.mean(self.episode_distances)
            maximum = np.max(self.episode_distances)
            
            # Log the metrics to TensorBoard under the "custom" tab
            self.logger.record("custom/average_distance", average)
            self.logger.record("custom/maximum_distance", maximum)
            
            # Clear the buffer for the next rollout
            self.episode_distances = []
            
# Environment factory function required for multiprocessing
def make_env():
    def _init():
        # 1. Create the base environment
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # 2. Image preprocessing
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)

        # 3. Apply the custom wrapper
        env = CustomEnvWrapper(env)
        return env
    return _init

# Main block strictly required for safe multiprocessing in Python
if __name__ == '__main__':
    save_dir = "./mario_models/"
    os.makedirs(save_dir, exist_ok=True)

    # 4. Vectorize the environment across multiple CPUs
    num_cpu = 10
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])
    env = VecFrameStack(env, n_stack=4)

    # 5. Monitor to track metrics in TensorBoard for vectorized environments
    env = VecMonitor(env)

    total_timesteps = 500000
    base_model_path = "./14-09_model_2832000_steps"

    print("--- Starting training across {} CPUs ---".format(num_cpu))

    CHECK_FREQ_NUMB = 10000
    TOTAL_TIMESTEP_NUMB = 5000000
    LEARNING_RATE = 0.0001
    GAE = 1.0 
    ENT_COEF = 0.01
    N_STEPS = 512
    GAMMA = 0.9
    BATCH_SIZE = 64
    N_EPOCHS = 10

    # Map the constants to the exact dictionary keys expected by PPO
    custom_objects = {
        "learning_rate": LEARNING_RATE,
        "gae_lambda": GAE,
        "ent_coef": ENT_COEF,
        "n_steps": N_STEPS,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS
    }

    print(f"Loading base model from: {base_model_path}")
    
    # Load the pre-trained model and override with the current parameters
    model = PPO.load(
        base_model_path,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./tensorboard_logs/"
    )

    # Configure automatic saving (checkpoints)
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=save_dir,
        name_prefix="15-09_model"
    )

    # Initialize the custom distance tracker
    distance_callback = DistanceTrackingCallback()

    # Group the callbacks into a list
    callback_list = CallbackList([checkpoint_callback, distance_callback])

    # Start training
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,          # <-- Pass the callback list here
        tb_log_name='15-09_model_total',
        reset_num_timesteps=False
    )
    
    # Save the final model after the run completes
    final_model_path = os.path.join(save_dir, "15-09_model_final")
    model.save(final_model_path)
    print(f"Model saved successfully: {final_model_path}\n")

    # Cleanly close the environments
    env.close()
