import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO

# ==========================================
# EVALUATION WRAPPER (No Max Pooling)
# ==========================================
class MarioEvalWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        
    def reset(self, **kwargs):
        obs = self.env.reset()
        
        # Handle Gym backwards compatibility (tuple unwrapping)
        if isinstance(obs, tuple):
            obs, reset_info = obs
        else:
            reset_info = {}
            
        self.env.render(mode="human")
        return obs, reset_info
        
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # Apply frame skipping while rendering each frame in real-time
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            
            self.env.render(mode="human")
            if done:
                break
                
        # Return the last frame directly without merging
        return obs, total_reward, done, False, info

# ==========================================
# ENVIRONMENT SETUP
# ==========================================
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = GrayScaleObservation(env, keep_dim=True)
env = ResizeObservation(env, shape=84)
env = MarioEvalWrapper(env, skip=4)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, n_stack=4)

# ==========================================
# MODEL LOADING & EXECUTION
# ==========================================
model_path = "./15-09_model_3252000_steps"
print(f"Loading model from {model_path}...")
model = PPO.load(model_path, env=env)

obs = env.reset()
print("Starting evaluation. Close the window to stop.")

while True:
    # deterministic=True forces the agent to exploit its learned policy rather than taking random actions
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

env.close()
