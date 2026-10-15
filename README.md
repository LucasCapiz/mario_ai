# Super Mario Bros AI: Mastering Level 1-1 with PPO

### Overview
This project trains a Reinforcement Learning (RL) agent to play and beat *Super Mario Bros* (NES) using the Proximal Policy Optimization (PPO) algorithm. The current model has successfully mastered World 1-1, achieving a 100% win rate through optimized reward shaping, computer vision preprocessing, and parallelized training environments.

### Tech Stack
*   **Algorithm:** Proximal Policy Optimization (PPO) via `stable-baselines3`
*   **Environment:** `gym-super-mario-bros` (OpenAI Gym wrapper for the NES emulator)
*   **Frameworks:** PyTorch, Python, NumPy
*   **Monitoring:** TensorBoard (with custom metric callbacks)

### Key Technical Implementations
*   **Custom Environment Wrapper:** Implemented Grayscale conversion, ResizeObservation (84x84), and Frame Skipping (skip=4) to reduce the state space and accelerate training.
*   **Multiprocessing:** Utilized `SubprocVecEnv` to distribute the environment across 10 independent CPU cores, drastically speeding up experience collection (totaling millions of timesteps).
*   **Custom Analytics:** Engineered a custom TensorBoard callback to bypass standard arbitrary reward metrics (`ep_rew_mean`) and track absolute performance indicators: **Win Rate** and **Maximum X-Distance** per episode.
*   **Hyperparameter Tuning:** Fine-tuned critical PPO parameters including learning rate decay, entropy coefficient (to balance exploration vs. exploitation), batch size, and `n_steps` for optimal gradient updates.

### Current Progress & Roadmap
*   **Phase 1:** Master World 1-1 (Completed - 100% Win Rate).
*   **Phase 2:** Generalization across World 1. Currently transitioning the architecture to train on Worlds 1-1 through 1-4 simultaneously to prevent level-overfitting and develop universal platforming intelligence.
*   **Phase 3:** Implement Curriculum Learning and Random Stage initialization (`SuperMarioBrosRandomStages-v0`) for zero-shot level completion.

<p align="center">
  <br>
  <img width="417" height="269" alt="_MConverter eu_mario_ai_video_1-1 (2) (2) (1)" src="https://github.com/user-attachments/assets/1234ae7e-a3be-4225-ac93-e81b8da94f27" />
  <br>
  <em>PPO Agent mastering Super Mario Bros World 1-1</em>
</p>
