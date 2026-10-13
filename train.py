import os
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation, ResizeObservation
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecMonitor
# from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3 import PPO
import numpy as np

# Envoltorio (Wrapper) para adaptar la compatibilidad de variables y aplicar Frame Skip
class CustomEnvWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        
    def reset(self, **kwargs):
        # Reinicio clasico del entorno
        obs = self.env.reset()
        
        # Manejo seguro de la tupla de compatibilidad de Gym
        if isinstance(obs, tuple):
            obs, info = obs
        else:
            info = {}
            
        return obs, info
        
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # Repetir la misma accion 'skip' veces para acelerar el aprendizaje (Frame Skip)
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        # Moldeado de recompensas (Reward Shaping):
        # Penalizar levemente si el agente se queda quieto o atascado en un obstaculo
        if total_reward <= 0:
            total_reward -= 0.1
            
        # Devolver los 5 valores esperados por la API de Gym requerida por PPO
        return obs, total_reward, done, False, info

class DistanceTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_distances = []

    def _on_step(self) -> bool:
        # Revisa la información de los entornos vectorizados en cada paso
        for info in self.locals.get("infos", []):
            if "x_pos" in info:
                # Si el jugador muere o llega a la bandera, guardamos su x_pos final
                if "terminal_observation" in info:
                    self.episode_distances.append(info["x_pos"])
        return True

    def _on_rollout_end(self) -> None:
        # Se ejecuta cada vez que el PPO actualiza sus redes (ej. cada n_steps * num_envs)
        if len(self.episode_distances) > 0:
            average = np.mean(self.episode_distances)
            maximum = np.max(self.episode_distances)
            
            # Escribe las métricas en TensorBoard bajo la pestaña "custom"
            self.logger.record("custom/average_distance", average)
            self.logger.record("custom/maximum_distance", maximum)
            
            # Limpiamos para el próximo ciclo
            self.episode_distances = []
            
# Funcion constructora necesaria para instanciar entornos independientes en cada CPU
def make_env():
    def _init():
        # 1. Creacion del entorno base
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

        # 2. Preprocesamiento de imagenes
        env = GrayScaleObservation(env, keep_dim=True)
        env = ResizeObservation(env, shape=84)

        # 3. Aplicacion del wrapper personalizado
        env = CustomEnvWrapper(env)
        return env
    return _init

# Bloque principal obligatorio para multiprocesamiento seguro en Python
if __name__ == '__main__':
    save_dir = "./mario_models/"
    os.makedirs(save_dir, exist_ok=True)

    # 4. Vectorizacion para el algoritmo con multiples CPUs
    num_cpu = 10
    env = SubprocVecEnv([make_env() for i in range(num_cpu)])
    env = VecFrameStack(env, n_stack=4)

    # 5. Monitor para registrar metricas en TensorBoard adaptado a entornos vectorizados
    env = VecMonitor(env)

    total_timesteps = 500000
    base_model_path = "./mario_models/14-09_model_2832000_steps"

    print("--- Iniciando entrenamiento con {} CPUs ---".format(num_cpu))

    # Mantenemos los hiperparametros solicitados en un diccionario 
    # para forzarlos al cargar el modelo existente
    # custom_objects = {
    #     "learning_rate": 0.0001,
    #     "ent_coef": 0.0001,
    #     "n_steps": 1024,
    #     "batch_size": 1024
    # }

    CHECK_FREQ_NUMB = 10000
    TOTAL_TIMESTEP_NUMB = 5000000
    LEARNING_RATE = 0.0001
    GAE = 1.0 
    # 0.95
    ENT_COEF = 0.01
    N_STEPS = 512
    GAMMA = 0.9
    # gamma 0.99
    BATCH_SIZE = 64
    N_EPOCHS = 10

    # Mapeamos tus constantes al diccionario que PPO requiere
    custom_objects = {
        "learning_rate": LEARNING_RATE,
        "gae_lambda": GAE,          # En SB3 el parámetro se llama gae_lambda
        "ent_coef": ENT_COEF,
        "n_steps": N_STEPS,
        "gamma": GAMMA,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS
    }

    print(f"Cargando modelo base: {base_model_path}")
    
    # Carga del modelo previamente entrenado aplicando los parametros actuales
    model = PPO.load(
        base_model_path,
        env=env,
        custom_objects=custom_objects,
        tensorboard_log="./tensorboard_logs/"
    )

    # Configuracion de guardado automatico (checkpoints) actualizados al nuevo dia
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,
        save_path=save_dir,
        name_prefix="15-09_model"
    )

    # Inicializamos nuestro tracker de distancia
    distance_callback = DistanceTrackingCallback()

    # Agrupamos ambos callbacks
    callback_list = CallbackList([checkpoint_callback, distance_callback])

    # Entrenamiento del modelo
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,          # <-- Cambiamos esto para usar la lista
        tb_log_name='15-09_model_total',
        reset_num_timesteps=False
    )
    
    # Guardar el modelo final de la ejecucion con el nuevo nombre
    final_model_path = os.path.join(save_dir, "15-09_model_final")
    model.save(final_model_path)
    print(f"Modelo guardado exitosamente: {final_model_path}\n")

    # Cierre ordenado del entorno
    env.close()