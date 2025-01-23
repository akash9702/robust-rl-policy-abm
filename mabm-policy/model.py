from juliacall import Main as jl

from VanillaGov import VanillaGovABC
from Simulator import AdvGovABCSimulator, ARLABCSimulator
from GovAdvEnvs import GovABC, AdvABC, ARLAdv, ARLGov

import gymnasium as gym

from stable_baselines3 import PPO, SAC, TD3, A2C, DDPG
from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise


import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



import numpy as np
import sys
import time
import json
import csv
import os
import gc
import re
import threading
from copy import deepcopy
import torch

def compute_batch_size(n: int) -> int:
    """
    not very scientific
    we want at least 3 updates per rollout => return batch_size <= n/3
    we want at most 10 updates per rollouts => return batch_size >= n/10
    out of these, we want the least number of updates 
    """
    for factor in range(3, 11):
        if n % factor == 0:
            return int(n / factor)
    return n
    


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    This callback should:
        1) save all checkpoints (models, envs, logs, data(?)) regularly
        2) save the best model (and its corresponding env) after every freq steps (episodes?)
        3) save the validation rewards and log them too

    assumptions:
        - freq = n_steps = ep_len? 
    
    TODO:
        - should we take ep_len and n_eps_before_update as params instead of freq?
        - can this be used for both vanilla and rarl?
        - given that verbose = 1, print more stuff

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    args:
        ep_len: length of episode
        update_freq: number of episodes after which model updated
        save_path: the path to save stuff (models, data, logs, envs)
        verbose: verbosity level
        test_env: a Vanilla instance with the training config (should be a copy of the train env in Vanilla callback)
    """
    def __init__(self, ep_len: int, update_freq: int, save_path: str, verbose: int, test_env, n_samples: int = 3):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.n_samples = n_samples

        os.makedirs(name = save_path, exist_ok = True)

        self.model_dir = os.path.join(save_path, "models")
        os.makedirs(name = self.model_dir, exist_ok = True)
        self.log_dir = os.path.join(save_path, "logs")
        os.makedirs(name = self.log_dir, exist_ok = True)
        self.env_dir = os.path.join(save_path, "envs")
        os.makedirs(name = self.env_dir, exist_ok = True)
        self.data_dir = os.path.join(save_path, "data")
        os.makedirs(name = self.data_dir, exist_ok = True)

        self.ep_len = ep_len
        self.update_freq = update_freq

        # we should save everything after every update, because that's when the model weights will actually change
        self.freq = int(ep_len * update_freq)

        self.test_env = test_env
        
        self.best_val_reward = -np.inf


        self.iter = 0
        # given that while loading the ModelManager, the callback will be re-instantiated, we need to set iter for model and env naming 
        model_files = os.listdir(self.model_dir)
        number_pattern = re.compile(r"^(\d+)\.zip$")
        for file in model_files:
            match = number_pattern.match(file)
            if match:
                this_iter = int(match.group(1))
                if this_iter > self.iter:
                    self.iter = this_iter

        
        # self.flag = False
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        needs access to
            - the current env for saving it
            - the iteration of model

        we are using this instead of on_rollout_end since for some algorithms updates might not happen just after rollouts
        although TODO we should think more about whether we should do it on rollout end

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        
        # if self.flag:
        #     self.episode_reward += self.training_env.get_original_reward().item()
        #     self.episode_norm_reward += self.training_env.normalize_reward(self.training_env.get_original_reward().item())

        # if (self.num_timesteps + self.n_samples * self.ep_len) % self.freq == 0:
        #     self.flag = True
        #     self.episode_reward = 0
        #     self.episode_norm_reward = 0


        # if self.n_calls % self.freq == 0:
        if self.num_timesteps % self.freq == 0:
            self.iter += 1

            # self.flag = False
            
            # ep_unnorm_rew_mean = self.episode_reward / self.n_samples
            
            # ep_norm_rew_mean = self.episode_norm_reward / self.n_samples

            # self.logger.record("rollout/ep_unnorm_rew_mean", ep_unnorm_rew_mean)
            # self.logger.record("rollout/ep_norm_rew_mean", ep_norm_rew_mean)

            # if isinstance(self.model, PPO):
            #     self.logger.record("rollout/ent_coef", self.model.ent_coef)

            

            # self.model.save(os.path.join(self.model_dir, f"{self.iter}.zip"))
            
            # # save the env only if it's of type VecNormalize (otherwise useless)
            # if isinstance(self.training_env, VecNormalize):
            #     self.training_env.save(os.path.join(self.env_dir, f"{self.iter}.pkl"))

            # print and save the logs if verbose = 1? TODO

            # save the data? TODO
            # what data to save? 
            #   - test env reward after each update (becasue it should stay roughly constant across episodes for one un-updated model)
            #   - after each training, it needs to save the graph of test_env rewards  
            #       if a model is loaded and trained again, it can be made and saved again (overwriting the last one), since we would have access to the file containing the earlier test env rewards for each episode
            
            # validation_reward = self._eval_on_test(n_samples = self.n_samples)

            # # open csv file containing everything (create it if it doesn't exist)
            # with open(file = os.path.join(self.data_dir, "rollout_test_rewards.csv"), mode = "a+", newline = '') as csvfile:
            #     writer = csv.writer(csvfile)

            #     writer.writerow([validation_reward])

            # # log the validation reward to tensorboard
            # self.logger.record("rollout/ep_val_rew", validation_reward)

            # # save best model
            # if validation_reward > self.best_val_reward:
                
            #     self.best_val_reward = validation_reward

            #     self.model.save(os.path.join(self.model_dir, "best_model.zip"))

            #     # save env (only if it's VecNormalize)
            #     if isinstance(self.training_env, VecNormalize):
            #         self.training_env.save(os.path.join(self.env_dir, "best_model_env.pkl"))

            # # ep_norm_val_rew = self.training_env.normalize_reward(validation_reward)
            # # self.logger.record("rollout/ep_norm_val_rew", ep_norm_val_rew)

            
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """

        # plot rewards

        # with open(os.path.join(self.data_dir, "rollout_test_rewards.csv"), mode = "r", newline = "") as csvfile:
        #     reader = csv.reader(csvfile)
            
        #     # we know it has 1 value per row
        #     rollout_test_rewards = np.array([float(row[0]) for row in reader], dtype = np.float32)
        
        # # since rollout test rewards are stored after every self.update_freq episodes
        # episodes_axis = np.array(
        #     [
        #         self.update_freq * (iter + 1) for iter in range(rollout_test_rewards.shape[0])
        #     ], 
        #     dtype = np.float32
        # )

        # plt.plot(episodes_axis, rollout_test_rewards, label = "episodic reward")
        # plt.xlabel("episodes")
        # plt.ylabel("reward")
        # plt.savefig(os.path.join(self.data_dir, "episodic_reward.png"))
        # plt.close()

    def _eval_on_test(self, n_samples: int = 5) -> float:
        """
        TODO: this should also be done in VanillaCallback since 
            1) it stores the average over n_samples episodes
            2) it takes the deterministic policy => it is the correct validation reward
        args:
            - n_samples: number of episodes to sample
        """

        ep_rew = 0

        for _ in range(n_samples):

            obs, info = self.test_env.reset()

            for i in range(self.ep_len):
                
                # need to normalize obs to use model.predict
                if isinstance(self.training_env, VecNormalize):
                    norm_obs = self.training_env.normalize_obs(obs)
                else:
                    norm_obs = obs

                action, _ = self.model.predict(norm_obs, deterministic = True)

                obs, reward, terminated, truncated, info = self.test_env.step(action)

                ep_rew += reward
        
        ep_rew /= n_samples

        self.test_env.reset()

        return ep_rew
    


class VanillaModelManager:

    """
    args:
        - env_config: contains the entire (or partial) config of the environment -- should always be consistent with defaults if needed
        - env_class: class of env (VanillaGovABC, etc) -- TODO: why is this needed?
        - update_freq: number of episodes before update
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",
        policy_type: str = "MlpPolicy",
        env_config = {},
        env_class = VanillaGovABC,
        additional: dict = {"callback_n_samples" : 1},
        update_freq: int = 2,
        n_envs: int = 1,
        batch_size: int = None,
        dir: str = None,
        seed: int = 42
    ):
        self.algorithm = algorithm.lower()
        self.policy_type = policy_type
        
        self.env_config = env_config
        # self.train_env = VanillaGovABC(**env_config)
        self.env_class = env_class
        self.train_env = self.env_class(**self.env_config)
        self.env_config = self.train_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq
        
        self.episode_length = int(self.train_env.T / self.train_env.n_const_steps)

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-7:] != "Vanilla" and self.dir[-7:] != "vanilla":
            self.dir = os.path.join(self.dir, "Vanilla")

        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = seed

        self.model_dir = os.path.join(self.dir, "models")
        self.env_dir = os.path.join(self.dir, "envs")

        # set_random_seed(self.seed)

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.train_env),
            n_samples = self.additional["callback_n_samples"],
        )

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003
        
        # if "schedule" in self.additional:
        #     if self.additional["schedule"] == "linear":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         final_lr = self.additional.get("final_lr", 3e-6)
        #         self.learning_rate = lambda progress: initial_lr + (1 - progress) * (final_lr - initial_lr)
        #     elif self.additional["schedule"] == "exponential":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         decay_rate = self.additional.get("decay_rate", 0.5)
        #         self.learning_rate = lambda progress: initial_lr * (decay_rate ** (1 - progress))
        #     elif self.additional["schedule"] == "cosine":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         min_lr = self.additional.get("min_lr", 1e-4)
        #         self.learning_rate = lambda progress: min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * (1 - progress)))
                

    def make(
        self,
        norm_obs: bool = True,
        norm_reward: bool = True
    ):
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        # make env
        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)
        if norm_obs or norm_reward:
            self.env = VecNormalize(self.env, norm_obs = norm_obs, norm_reward = norm_reward)

        ent_coef = 0.01
        if "ent_coef" in self.additional:
            ent_coef = self.additional["ent_coef"]
    

        if self.algorithm == "ppo":
            self.model = PPO(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1, device = "cpu",
                seed = self.seed,
                ent_coef = ent_coef
            )

        elif self.algorithm == "trpo":
            self.model = TRPO(
                policy=self.policy_type,
                env=self.env,
                learning_rate=self.learning_rate,
                target_kl=0.01,
                n_steps = self.n_steps,
                verbose=1, device = "cpu",
                seed=self.seed
            )

        elif self.algorithm == "a2c":
            self.model = A2C(
                policy=self.policy_type,
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps // 10,
                verbose=1, device = "cpu",
                seed=self.seed,
                ent_coef=ent_coef
            )

        
        elif self.algorithm == "sac":
            self.model = SAC(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1, device = "cpu",
                seed = self.seed
            )
            # self.model = SAC(
            #     policy = self.policy_type,
            #     env = self.env,
            #     learning_rate = 5e-5,
            #     train_freq = self.n_steps,
            #     batch_size = self.batch_size,
            #     verbose = 1, device = "cpu",
            #     ent_coef=0.2,
            #     seed = self.seed
            # )
        elif self.algorithm == "td3":
            
            output_dim = self.env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )
            
            self.model = TD3(
                policy=self.policy_type,
                env=self.env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.05,
                target_noise_clip=0.2,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.algorithm == "ddpg":
            
            output_dim = self.env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )

            self.model = DDPG(
                policy=self.policy_type,
                env=self.env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise = action_noise,
                verbose=1, device = "cpu",
                seed = self.seed
            )

        self.model.set_logger(
            configure(
                folder = self.log_dir,
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )
        
        self.save()


    def load(
        self
    ):
        self.save_dir = os.path.join(self.dir, "saved")

        # with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
        #     self.config = json.load(file)

        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)

        # normalizing if required
        if os.path.exists(os.path.join(self.save_dir, "env.pkl")):
            self.env = VecNormalize.load(load_path = os.path.join(self.save_dir, "env.pkl"), venv = self.env)

        if self.algorithm == "ppo":
            self.model = PPO.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "sac":
            self.model = SAC.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "td3":
            # TODO: check whether to load with noise
            self.model = TD3.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "ddpg":
            # TODO: check whether to load with noise
            self.model = DDPG.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "trpo":
            self.model = TRPO.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)


        self.model.set_logger(
            configure(
                folder = self.log_dir,
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.save()


    def save(
        self
    ):
        """
        TODO:
            - should save current model and current env and all other things
        """
        self.save_dir = os.path.join(self.dir, "saved")
        os.makedirs(self.save_dir, exist_ok = True)

        # need to remove keys that aren't JSON serializable
        # TODO: add to this
        keys_to_remove = {
            "model",
            "train_env",
            "env",
            "callback",
            "learning_rate",
            "env_class",
            "episode_length",
            "n_steps",
            "log_dir",
            "norm_obs",
            "norm_reward",
            "save_dir",
            "config",
            "model_dir",
            "env_dir"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)
                if type(value) in [np.int32, np.int64, np.int16]:
                    self.config[key] = int(value)
                if type(value) == dict:
                    copied_value = deepcopy(value)
                    for nested_key, nested_value in copied_value.items():
                        if type(nested_value) in [np.int32, np.int64, np.int16]:
                            copied_value[nested_key] = int(nested_value)
                    self.config[key] = copied_value

        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.model.save(os.path.join(self.save_dir, "model.zip"))

        if isinstance(self.env, VecNormalize):
            # self.env.save(os.path.join(self.save_dir, "env.pkl"))
            self.model.get_vec_normalize_env().save(os.path.join(self.save_dir, "env.pkl"))

    def close_threads(self):

        self.model.env.close()

        # After training, flush and close TensorBoard writers
        for output_format in self.model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()


    def train_stop(
        self,
        N_iter,
        N_mu
    ):
        # load the model before every call of train_stop (good measure)
        self.load()

        total_timesteps = N_mu * self.n_steps
        iter = 0
        while iter < N_iter:
            iter += 1
            self.model.learn(
                total_timesteps = total_timesteps,
                reset_num_timesteps = False,
                log_interval = 1,
                tb_log_name = "Vanilla"
            )
            # if iter % (N_iter // 10) == 0:
            if (iter - 1) % 10 == 0:

                self.model.save(os.path.join(self.model_dir, f"{iter}.zip"))
                
                # save the env only if it's of type VecNormalize (otherwise useless)
                if isinstance(self.model.env, VecNormalize):
                    self.model.env.save(os.path.join(self.env_dir, f"{iter}.pkl"))
            
            torch.cuda.empty_cache()
            gc.collect()
                
        self.save()
        self.close_threads()

        del self.model



    def train_no_stop(
        self,
        N_mu
    ):
        total_timesteps = self.n_steps * N_mu
        while True:
            self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, log_interval = 1, tb_log_name="Vanilla", callback = self.callback)



"""
1e5
2e5
"""
        

class RARLModelManager:

    """
    args:
        - env_config: contains the entire (or partial) config of the environment -- should always be consistent with defaults if needed
        - env_class: class of env (VanillaGovABC, etc) -- TODO: why is this needed?
        - update_freq: number of episodes before update
    """
    
    def __init__(
        self,
        gov_algorithm: str = "PPO",
        adv_algorithm: str = "PPO",
        gov_policy_type: str = "MlpPolicy",
        adv_policy_type: str = "MlpPolicy",
        sim_config = {},
        vanilla_class = VanillaGovABC,
        sim_class = AdvGovABCSimulator,
        gov_class = GovABC,
        adv_class = AdvABC,
        additional: dict = {"callback_n_samples" : 1},
        update_freq: int = 2,
        n_envs: int = 1,
        batch_size: int = None,
        dir: str = None,
        seed: int = 42
    ):
        self.gov_algorithm = gov_algorithm.lower()
        self.gov_policy_type = gov_policy_type
        self.adv_algorithm = adv_algorithm.lower()
        self.adv_policy_type = adv_policy_type

        self.sim_class = sim_class
        self.gov_class = gov_class
        self.adv_class = adv_class
        self.vanilla_class = vanilla_class
        
        self.sim_config = sim_config
        
        self.simulator = self.sim_class(**self.sim_config)

        self.sim_config = self.simulator.get_config()

        # train env is used to denote non vectorized env
        self.gov_train_env = self.gov_class(self.simulator)
        self.adv_train_env = self.adv_class(self.simulator)
        
        self.env_config = deepcopy(self.sim_config)

        self.env_config["action_types"] = self.env_config["gov_action_types"]
        del self.env_config["gov_action_types"]
        del self.env_config["adv_action_types"]
        self.env_config["observation_types"] = self.env_config["gov_observation_types"]
        del self.env_config["gov_observation_types"]
        del self.env_config["adv_observation_types"]

        self.vanilla_env = self.vanilla_class(**self.env_config)

        self.env_config = self.vanilla_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq

        self.episode_length = int(self.simulator.T / self.simulator.n_const_steps)

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-4:] != "RARL" and self.dir[-4:] != "rarl":
            # print("printing inside RARLModelManager")
            # print(self.dir)
            self.dir = os.path.join(self.dir, "RARL")

        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = seed
        
        self.model_dir = os.path.join(self.dir, "models")
        self.env_dir = os.path.join(self.dir, "envs")

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = self.update_freq,
            save_path = self.dir,
            verbose = 1, 
            test_env = deepcopy(self.vanilla_env),
            n_samples = self.additional["callback_n_samples"]
        )

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003
        # if "schedule" in self.additional:
        #     if self.additional["schedule"] == "linear":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         final_lr = self.additional.get("final_lr", 3e-6)
        #         self.learning_rate = lambda progress: initial_lr + (1 - progress) * (final_lr - initial_lr)
        #     elif self.additional["schedule"] == "exponential":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         decay_rate = self.additional.get("decay_rate", 0.5)
        #         self.learning_rate = lambda progress: initial_lr * (decay_rate ** (1 - progress))
        #     elif self.additional["schedule"] == "cosine":
        #         initial_lr = self.additional.get("initial_lr", 3e-4)
        #         min_lr = self.additional.get("min_lr", 1e-4)
        #         self.learning_rate = lambda progress: min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * (1 - progress)))
                
        # self.learning_rate = lambda progress: 0.0003 + (3e-5 - 3e-4) * (1 - progress)

    def make(
        self,
        norm_obs: bool = True,
        norm_reward: bool = True
    ):
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        # make gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)
        if norm_obs or norm_reward:
            self.gov_env = VecNormalize(self.gov_env, norm_obs = self.norm_obs, norm_reward = self.norm_reward)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        if norm_obs or norm_reward:
            self.adv_env = VecNormalize(self.adv_env, norm_obs = self.norm_obs, norm_reward = self.norm_reward)


        ent_coef = 0.01
        if "ent_coef" in self.additional:
            ent_coef = self.additional["ent_coef"]
        
        
        if self.gov_algorithm == "ppo":

            self.gov_model = PPO(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1, device = "cpu",
                seed = self.seed,
                ent_coef = ent_coef
            )

        
        elif self.gov_algorithm == "trpo":
            self.gov_model = TRPO(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                target_kl=0.01,
                n_steps = self.n_steps,
                verbose=1, device = "cpu",
                seed=self.seed
            )

        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps // 10,
                verbose=1, device = "cpu",
                seed=self.seed,
                ent_coef=ent_coef
            )
        
      
        elif self.gov_algorithm == "td3":
            
            output_dim = self.gov_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )
            
            self.gov_model = TD3(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.05,
                target_noise_clip=0.2,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.gov_algorithm == "ddpg":
            
            output_dim = self.gov_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )

            self.gov_model = DDPG(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise = action_noise,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.gov_algorithm == "sac":


            self.gov_model = SAC(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1, device = "cpu",
                seed = self.seed
            )
        
        if self.adv_algorithm == "ppo":

            self.adv_model = PPO(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1, device = "cpu",
                seed = self.seed,
                ent_coef = ent_coef
            )
        elif self.adv_algorithm == "sac":
            self.adv_model = SAC(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1, device = "cpu",
                seed = self.seed
            )      
        elif self.adv_algorithm == "trpo":
            self.adv_model = TRPO(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                target_kl=0.01,
                n_steps = self.n_steps,
                verbose=1, device = "cpu",
                seed=self.seed
            )

        elif self.adv_algorithm == "a2c":
            self.adv_model = A2C(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps // 10,
                verbose=1, device = "cpu",
                seed=self.seed,
                ent_coef=ent_coef
            )
        
      
        elif self.adv_algorithm == "td3":
            
            output_dim = self.adv_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )
            
            self.adv_model = TD3(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.05,
                target_noise_clip=0.2,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.adv_algorithm == "ddpg":
            
            output_dim = self.adv_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )

            self.adv_model = DDPG(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise = action_noise,
                verbose=1, device = "cpu",
                seed = self.seed
            )

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "RARL"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.adv_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "Adv"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.save()



    def load(
        self
    ):
        self.save_dir = os.path.join(self.dir, "saved")

        with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
            self.config = json.load(file)

        # load gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)
        if os.path.exists(os.path.join(self.save_dir, "gov_env.pkl")):
            self.gov_env = VecNormalize.load(load_path = os.path.join(self.save_dir, "gov_env.pkl"), venv = self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        if os.path.exists(os.path.join(self.save_dir, "adv_env.pkl")):
            self.adv_env = VecNormalize.load(load_path = os.path.join(self.save_dir, "adv_env.pkl"), venv = self.adv_env)

        if self.gov_algorithm == "ppo":
            self.gov_model = PPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "sac":
            self.gov_model = SAC.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = SAC.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "td3":
            # TODO: check whether to load with noise
            self.gov_model = TD3.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = TD3.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "ddpg":
            # TODO: check whether to load with noise
            self.gov_model = DDPG.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = DDPG.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = A2C.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "trpo":
            self.gov_model = TRPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = TRPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "RARL"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.adv_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "Adv"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.save()

    def save(
        self
    ):
        """
        TODO:
            - should save current model and current env and all other things
        """
        self.save_dir = os.path.join(self.dir, "saved")
        os.makedirs(self.save_dir, exist_ok = True)

        # self.config = deepcopy(vars(self))

        # need to remove keys that aren't JSON serializable
        # TODO: add to this
        keys_to_remove = {
            "gov_model",
            "adv_model",
            "simulator",
            "gov_train_env",
            "adv_train_env",
            "vanilla_env",
            "gov_env",
            "adv_env",
            "callback",
            "learning_rate",
            "vanilla_class",
            "sim_class",
            "gov_class",
            "adv_class",
            "episode_length",
            "n_steps",
            "log_dir",
            "norm_obs",
            "norm_reward",
            "save_dir",
            "config",
            "env_config",
            "model_dir",
            "env_dir"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)
                if type(value) in [np.int32, np.int64, np.int16]:
                    self.config[key] = int(value)
                if type(value) == dict:
                    copied_value = deepcopy(value)
                    for nested_key, nested_value in copied_value.items():
                        if type(nested_value) in [np.int32, np.int64, np.int16]:
                            copied_value[nested_key] = int(nested_value)
                    self.config[key] = copied_value

        # for key in keys_to_remove:
        #     if key in self.config:
        #         del self.config[key]

        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.gov_model.save(os.path.join(self.save_dir, "gov_model.zip"))

        if isinstance(self.gov_env, VecNormalize):
            # self.gov_env.save(os.path.join(self.save_dir, "gov_env.pkl"))
            self.gov_model.get_vec_normalize_env().save(os.path.join(self.save_dir, "gov_env.pkl"))

        self.adv_model.save(os.path.join(self.save_dir, "adv_model.zip"))

        if isinstance(self.adv_env, VecNormalize):
            # self.adv_env.save(os.path.join(self.save_dir, "adv_env.pkl"))
            self.adv_model.get_vec_normalize_env().save(os.path.join(self.save_dir, "adv_env.pkl"))

    def close_threads(
        self
    ):
        self.gov_model.env.close()
        self.adv_model.env.close()


        # After training, flush and close TensorBoard writers
        for output_format in self.gov_model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()
        

        for output_format in self.adv_model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()


    def train_stop(
        self,
        N_iter,
        N_mu,
        N_nu: int = None
    ):
        self.load()

        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = self.n_steps * N_nu

        iter = 0
        while iter < N_iter:
            iter += 1

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "RARL")

            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")


            if iter % (N_iter // 10) == 0:

                self.gov_model.save(os.path.join(self.model_dir, f"{iter}.zip"))
                
                # save the env only if it's of type VecNormalize (otherwise useless)
                if isinstance(self.gov_model.env, VecNormalize):
                    self.gov_model.env.save(os.path.join(self.env_dir, f"{iter}.pkl"))

            torch.cuda.empty_cache()
            gc.collect()

        self.save()
        self.close_threads()
            
        del self.gov_model
        del self.adv_model  
        torch.cuda.empty_cache()
        gc.collect()

    def train_no_stop(
        self,
        N_mu,
        N_nu: int = None
    ):
        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = self.n_steps * N_nu

        while True:

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "RARL", callback = self.callback)

            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")



class ARLModelManager:

    """
    args:
        - env_config: contains the entire (or partial) config of the environment -- should always be consistent with defaults if needed
        - env_class: class of env (VanillaGovABC, etc) -- TODO: why is this needed?
        - update_freq: number of episodes before update
    """
    
    def __init__(
        self,
        gov_algorithm: str = "PPO",
        adv_algorithm: str = "PPO",
        gov_policy_type: str = "MlpPolicy",
        adv_policy_type: str = "MlpPolicy",
        sim_config = {},
        vanilla_class = VanillaGovABC,
        sim_class = ARLABCSimulator,
        gov_class = ARLGov,
        adv_class = ARLAdv,
        additional: dict = {"callback_n_samples" : 1},
        update_freq: int = 2,
        n_envs: int = 1,
        batch_size: int = None,
        dir: str = None,
        seed: int = 42
    ):
        self.gov_algorithm = gov_algorithm.lower()
        self.gov_policy_type = gov_policy_type
        self.adv_algorithm = adv_algorithm.lower()
        self.adv_policy_type = adv_policy_type

        self.sim_class = sim_class
        self.gov_class = gov_class
        self.adv_class = adv_class
        self.vanilla_class = vanilla_class
        
        self.sim_config = sim_config
        
        self.simulator = self.sim_class(**self.sim_config)

        self.sim_config = self.simulator.get_config()

        # train env is used to denote non vectorized env
        self.gov_train_env = self.gov_class(self.simulator)
        self.adv_train_env = self.adv_class(self.simulator)
        
        self.env_config = deepcopy(self.sim_config)

        self.env_config["action_types"] = self.env_config["gov_action_types"]
        del self.env_config["gov_action_types"]
        del self.env_config["adv_action_types"]
        self.env_config["observation_types"] = self.env_config["gov_observation_types"]
        del self.env_config["gov_observation_types"]
        del self.env_config["adv_observation_types"]

        self.vanilla_env = self.vanilla_class(**self.env_config)

        self.env_config = self.vanilla_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq

        self.episode_length = int(self.simulator.T / self.simulator.n_const_steps)

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-3:].lower() != "arl":
            self.dir = os.path.join(self.dir, "ARL")

        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = seed

        self.model_dir = os.path.join(self.dir, "models")
        self.env_dir = os.path.join(self.dir, "envs")

        # set_random_seed(self.seed)

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = self.update_freq,
            save_path = self.dir,
            verbose = 1, 
            test_env = deepcopy(self.vanilla_env),
            n_samples = self.additional["callback_n_samples"]
        )

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003
                

    def make(
        self,
        norm_obs: bool = True,
        norm_reward: bool = True
    ):
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward

        # make gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)
        if norm_obs or norm_reward:
            self.gov_env = VecNormalize(self.gov_env, norm_obs = self.norm_obs, norm_reward = self.norm_reward)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        if norm_obs or norm_reward:
            self.adv_env = VecNormalize(self.adv_env, norm_obs = self.norm_obs, norm_reward = self.norm_reward)

        ent_coef = 0.01
        if "ent_coef" in self.additional:
            ent_coef = self.additional["ent_coef"]
        
        # if self.gov_algorithm == "ppo":

        #     self.gov_model = PPO(
        #         policy = self.gov_policy_type,
        #         env = self.gov_env,
        #         learning_rate = self.learning_rate,
        #         n_steps = self.n_steps,
        #         batch_size = self.batch_size,
        #         clip_range = 0.2,
        #         verbose = 1, device = "cpu",
        #         seed = self.seed,
        #         ent_coef = ent_coef
        #     )

        # elif self.gov_algorithm == "sac":


        #     self.gov_model = SAC(
        #         policy = self.gov_policy_type,
        #         env = self.gov_env,
        #         learning_rate = self.learning_rate,
        #         train_freq = self.n_steps,
        #         batch_size = self.batch_size,
        #         verbose = 1, device = "cpu",
        #         seed = self.seed
        #     )

            
        # elif self.gov_algorithm == "td3":
        #     pass
        # elif self.gov_algorithm == "ddpg":
        #     pass

        # if self.adv_algorithm == "ppo":

        #     self.adv_model = PPO(
        #         policy = self.adv_policy_type,
        #         env = self.adv_env,
        #         learning_rate = self.learning_rate,
        #         n_steps = int(self.n_steps / self.episode_length),
        #         batch_size = 2,
        #         clip_range = 0.2,
        #         verbose = 1, device = "cpu",
        #         seed = self.seed,
        #         ent_coef = ent_coef
        #     )
        # elif self.adv_algorithm == "sac":
        #     self.adv_model = SAC(
        #         policy = self.adv_policy_type,
        #         env = self.adv_env,
        #         learning_rate = self.learning_rate,
        #         train_freq = int(self.n_steps / self.episode_length),
        #         batch_size = 2,
        #         verbose = 1, device = "cpu",
        #         seed = self.seed
        #     )
        # elif self.adv_algorithm == "td3":
        #     pass
        # elif self.adv_algorithm == "ddpg":
        #     pass

         
        
        if self.gov_algorithm == "ppo":

            self.gov_model = PPO(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1, device = "cpu",
                seed = self.seed,
                ent_coef = ent_coef
            )

        
        elif self.gov_algorithm == "trpo":
            self.gov_model = TRPO(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                target_kl=0.01,
                n_steps = self.n_steps,
                verbose=1, device = "cpu",
                seed=self.seed
            )

        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps // 10,
                verbose=1, device = "cpu",
                seed=self.seed,
                ent_coef=ent_coef
            )
        
      
        elif self.gov_algorithm == "td3":
            
            output_dim = self.gov_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )
            
            self.gov_model = TD3(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.05,
                target_noise_clip=0.2,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.gov_algorithm == "ddpg":
            
            output_dim = self.gov_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )

            self.gov_model = DDPG(
                policy=self.gov_policy_type,
                env=self.gov_env,
                learning_rate=self.learning_rate,
                train_freq=self.n_steps,
                gradient_steps=1,
                action_noise = action_noise,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.gov_algorithm == "sac":


            self.gov_model = SAC(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1, device = "cpu",
                seed = self.seed
            )
        
        if self.adv_algorithm == "ppo":

            self.adv_model = PPO(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = int(self.n_steps / self.episode_length),
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1, device = "cpu",
                seed = self.seed,
                ent_coef = ent_coef
            )
        elif self.adv_algorithm == "sac":
            self.adv_model = SAC(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                train_freq = int(self.n_steps / self.episode_length),
                batch_size = self.batch_size,
                verbose = 1, device = "cpu",
                seed = self.seed
            )      
        elif self.adv_algorithm == "trpo":
            self.adv_model = TRPO(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                target_kl=0.01,
                n_steps = int(self.n_steps / self.episode_length),
                verbose=1, device = "cpu",
                seed=self.seed
            )

        elif self.adv_algorithm == "a2c":
            self.adv_model = A2C(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                n_steps=int(self.n_steps / self.episode_length),
                verbose=1, device = "cpu",
                seed=self.seed,
                ent_coef=ent_coef
            )
        
      
        elif self.adv_algorithm == "td3":
            
            output_dim = self.adv_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )
            
            self.adv_model = TD3(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                train_freq=int(self.n_steps / self.episode_length),
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.05,
                target_noise_clip=0.2,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        elif self.adv_algorithm == "ddpg":
            
            output_dim = self.adv_env.action_space.shape[0]
            noise_std = 0.2

            action_noise = NormalActionNoise(
                mean = np.zeros(output_dim),
                sigma = noise_std * np.ones(output_dim)
            )

            self.adv_model = DDPG(
                policy=self.adv_policy_type,
                env=self.adv_env,
                learning_rate=self.learning_rate,
                train_freq=int(self.n_steps / self.episode_length),
                gradient_steps=1,
                action_noise = action_noise,
                verbose=1, device = "cpu",
                seed = self.seed
            )


        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "ARL"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.adv_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "Adv"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.save()



    def load(
        self
    ):
        self.save_dir = os.path.join(self.dir, "saved")

        with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
            self.config = json.load(file)

        # load gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)
        if os.path.exists(os.path.join(self.save_dir, "gov_env.pkl")):
            self.gov_env = VecNormalize.load(load_path = os.path.join(self.save_dir, "gov_env.pkl"), venv = self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        if os.path.exists(os.path.join(self.save_dir, "adv_env.pkl")):
            self.adv_env = VecNormalize.load(load_path = os.path.join(self.save_dir, "adv_env.pkl"), venv = self.adv_env)

        if self.gov_algorithm == "ppo":
            self.gov_model = PPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "sac":
            self.gov_model = SAC.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = SAC.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "td3":
            # TODO: check whether to load with noise
            self.gov_model = TD3.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = TD3.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "ddpg":
            # TODO: check whether to load with noise
            self.gov_model = DDPG.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = DDPG.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = A2C.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "trpo":
            self.gov_model = TRPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = TRPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "ARL"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.adv_model.set_logger(
            configure(
                folder = os.path.join(self.log_dir, "Adv"),
                format_strings=["stdout", "csv", "tensorboard"]
            )
        )

        self.save()

    def save(
        self
    ):
        """
        TODO:
            - should save current model and current env and all other things
        """
        self.save_dir = os.path.join(self.dir, "saved")
        os.makedirs(self.save_dir, exist_ok = True)

        # self.config = deepcopy(vars(self))

        # need to remove keys that aren't JSON serializable
        # TODO: add to this
        keys_to_remove = {
            "gov_model",
            "adv_model",
            "simulator",
            "gov_train_env",
            "adv_train_env",
            "vanilla_env",
            "gov_env",
            "adv_env",
            "callback",
            "learning_rate",
            "vanilla_class",
            "sim_class",
            "gov_class",
            "adv_class",
            "episode_length",
            "n_steps",
            "log_dir",
            "norm_obs",
            "norm_reward",
            "save_dir",
            "config",
            "env_config",
            "model_dir",
            "env_dir"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)
                if type(value) in [np.int32, np.int64, np.int16]:
                    self.config[key] = int(value)
                if type(value) == dict:
                    copied_value = deepcopy(value)
                    for nested_key, nested_value in copied_value.items():
                        if type(nested_value) in [np.int32, np.int64, np.int16]:
                            copied_value[nested_key] = int(nested_value)
                    self.config[key] = copied_value

        # for key in keys_to_remove:
        #     if key in self.config:
        #         del self.config[key]

        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.gov_model.save(os.path.join(self.save_dir, "gov_model.zip"))

        if isinstance(self.gov_env, VecNormalize):
            self.gov_model.get_vec_normalize_env().save(os.path.join(self.save_dir, "gov_env.pkl"))

        self.adv_model.save(os.path.join(self.save_dir, "adv_model.zip"))

        if isinstance(self.adv_env, VecNormalize):
            self.adv_model.get_vec_normalize_env().save(os.path.join(self.save_dir, "adv_env.pkl"))


    def close_threads(
        self
    ):
        self.gov_model.env.close()
        self.adv_model.env.close()


        # After training, flush and close TensorBoard writers
        for output_format in self.gov_model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()
        

        for output_format in self.adv_model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()


    def train_stop(
        self,
        N_iter,
        N_mu,
        N_nu: int = None,
        close: bool = False
    ):
        
        self.load()

        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = int(N_nu * self.n_steps / self.episode_length)

        # n_steps = number of steps before an update = episode_length * update_freq = 1 * 2 = 2
        # adv_total_timesteps = update_freq * N_nu = 2 * 3 = 6
        # gov_total_timesteps = 6 * 300 = 1800

        iter = 0
        while iter < N_iter:
            iter += 1

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "ARL")
            
            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")


            if iter % (N_iter // 10) == 0:

                self.gov_model.save(os.path.join(self.model_dir, f"{iter}.zip"))
                
                # save the env only if it's of type VecNormalize (otherwise useless)
                if isinstance(self.gov_model.env, VecNormalize):
                    self.gov_model.env.save(os.path.join(self.env_dir, f"{iter}.pkl"))

            torch.cuda.empty_cache()
            gc.collect()
                

        self.save()
        self.close_threads()
            
        del self.gov_model
        del self.adv_model  
        torch.cuda.empty_cache()
        gc.collect()

        

    def train_stop_no_loop(
        self,
        N_iter,
        N_mu,
        N_nu = None
    ):
        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = int(N_nu * self.n_steps / self.episode_length)

        iter = 0
        while iter < N_iter / 10:
            iter += 1

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "ARL", callback = self.callback)
            
            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")
        
        iter = 0 
        
        self.gov_model.learn(total_timesteps = gov_total_timesteps * int(8 * N_iter / 10), reset_num_timesteps = False, log_interval = 1, tb_log_name = "ARL", callback = self.callback)

        self.adv_model.learn(total_timesteps = adv_total_timesteps * int(8 * N_iter / 10), reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")


        iter = 0
        while iter < N_iter / 10:
            iter += 1

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "ARL", callback = self.callback)
            
            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")

        self.save()

    def train_no_stop(
        self,
        N_mu,
        N_nu: int = None
    ):
        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = self.n_steps * N_nu

        while True:

            self.gov_model.learn(total_timesteps = gov_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "RARL", callback = self.callback)

            self.adv_model.learn(total_timesteps = adv_total_timesteps, reset_num_timesteps = False, log_interval = 1, tb_log_name = "Adv")


     
        

        
class ModelTester:

    def __init__(
        self,
        algo: str,
        dir: str,
        norm: bool,
        test_env
    ):
        algo = algo.lower()
        allowed_algos = ["ppo", "a2c", "sac", "td3", "ddpg", "trpo"]
        if not algo in allowed_algos:
            raise ValueError(f"algo must be in {str(allowed_algos)}")
            
        self.algo = algo
        
        if not (dir[-4:] == "RARL" or dir[-7:] == "Vanilla" or dir[-3:] == "ARL"):
            raise ValueError("vanilla or rarl or arl directory needs to be passed")
        self.dir = dir

        self.norm = norm

        self.model_dir = os.path.join(self.dir, "models")
        self.env_dir = os.path.join(self.dir, "envs")
        self.save_dir = os.path.join(self.dir, "saved")

        self.test_env = test_env

        self.episode_length = int(self.test_env.T / self.test_env.n_const_steps)

    def load_last(
        self
    ):
        is_vanilla = True
        if self.dir[-4:].lower() == "rarl" or self.dir[-3:].lower() == "arl":
            is_vanilla = False

        self.env = DummyVecEnv([lambda: deepcopy(self.test_env)])

        if is_vanilla:

            model_path = os.path.join(self.save_dir, "model.zip")
            if not os.path.exists(model_path):
                raise ValueError(f"model doesn't exist in {model_path}")

            if self.norm:
                self.env = VecNormalize.load(
                    load_path = os.path.join(self.save_dir, "env.pkl"),
                    venv = self.env
                )
        
        else:

            model_path = os.path.join(self.save_dir, "gov_model.zip")
            if not os.path.exists(model_path):
                raise ValueError(f"model doesn't exist in {model_path}")

            if self.norm:
                self.env = VecNormalize.load(
                    load_path = os.path.join(self.save_dir, "gov_env.pkl"),
                    venv = self.env
                )
            
        
        if self.algo.lower() == "ppo":
            self.model = PPO.load(
                path = model_path
            )
        elif self.algo.lower() == "sac":
            self.model = SAC.load(
                path = model_path
            )
        elif self.algo.lower() == "ddpg":
            self.model = DDPG.load(
                path = model_path
            )
        elif self.algo.lower() == "td3":
            self.model = TD3.load(
                path = model_path
            )
        elif self.algo.lower() == "a2c":
            self.model = A2C.load(
                path = model_path
            )
        elif self.algo.lower() == "trpo":
            self.model = TRPO.load(
                path = model_path
            )
            

    def load(
        self,
        checkpoint_no: int
    ):
        
        if not os.path.exists(os.path.join(self.model_dir, f"{checkpoint_no}.zip")):
            path_checked = os.path.join(self.model_dir, f"{checkpoint_no}.zip")
            error_details = f"model_dir: {self.model_dir} \n checkpoint_no: {checkpoint_no} \n path checked: {path_checked}"
            raise ValueError(f"model checkpoint doesn't exist\n{error_details}")
        
        if self.norm and not os.path.exists(os.path.join(self.env_dir, f"{checkpoint_no}.pkl")):
            raise ValueError(f"env checkpoint {checkpoint_no} doesn't exist")
        
        # load env

        self.env = DummyVecEnv([lambda: self.test_env])

        if self.norm:
            self.env = VecNormalize.load(
                load_path = os.path.join(self.env_dir, f"{checkpoint_no}.pkl"),
                venv = self.env
            )

        # load model

        if self.algo.lower() == "ppo":
            self.model = PPO.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo.lower() == "sac":
            self.model = SAC.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo.lower() == "ddpg":
            self.model = DDPG.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo.lower() == "td3":
            self.model = TD3.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo.lower() == "a2c":
            self.model = A2C.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo.lower() == "trpo":
            self.model = TRPO.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )

    def compute_reward(
        self,
        n_samples: int = 1
    ):
        ep_rew_mean = 0.0

        for i in range(n_samples):
            obs, info = self.test_env.reset()

            for j in range(self.episode_length):

                if self.norm:
                    norm_obs = self.env.normalize_obs(obs)
                else:
                    norm_obs = obs

                action, _ = self.model.predict(norm_obs, deterministic = True)

                obs, reward, term, trun, info = self.test_env.step(action)

                ep_rew_mean += reward

        ep_rew_mean /= n_samples

        self.test_env.reset()

        return ep_rew_mean
    
    def compute_reward_with_adversary(
        self,
        adv_model,
        adv_action_types,
        n_samples: int = 1
    ):
        ep_rew_mean = 0.0

        for i in range(n_samples):
            obs, info = self.test_env.reset()

            for j in range(self.episode_length):

                # we are assuming, what is true, that adv observation = gov observation always
                if isinstance(adv_model.env, VecNormalize):
                    norm_adv_obs = adv_model.env.normalize_obs(obs)
                else:
                    norm_adv_obs = obs

                adv_action, _ = adv_model.predict(norm_adv_obs, deterministic = True)
                
                # taking adversarial action
                action_np = np.array(adv_action, dtype = np.float32)
                for k in range(action_np.shape[0]):
                    
                    self.test_env.model.params[
                        jl.Symbol(self.test_env.actions_to_properties[adv_action_types[k]])
                    ] = adv_action[k]

                if self.norm:
                    norm_obs = self.env.normalize_obs(obs)
                else:
                    norm_obs = obs

                action, _ = self.model.predict(norm_obs, deterministic = True)

                obs, reward, term, trun, info = self.test_env.step(action)

                ep_rew_mean += reward

        ep_rew_mean /= n_samples

        self.test_env.reset()

        return ep_rew_mean

        


