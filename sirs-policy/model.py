from sirs_gov import SIRSGov, RARLSIRSAdv, RARLSIRSGov, RandSIRSGov, RARLSIRSAdv2, RARLSIRSGov2
from sirs_sim import SIRSSimulator, SIRSSimulator2
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np
from copy import deepcopy
import os
import json
import re
import dill as pickle
import csv
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import configure

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
    args:
        ep_len: length of episode
        update_freq: number of episodes after which model updated
        save_path: the path to save stuff (models, data, logs, envs)
        verbose: verbosity level
        test_env: a Vanilla instance with the training config (should be a copy of the train env in Vanilla callback)
    """
    def __init__(self, ep_len: int, update_freq: int, save_path: str, verbose: int, test_env, n_samples: int = 1, eval: bool = False):
        super().__init__(verbose)

        self.n_samples = n_samples
        self.eval = eval

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

        # if self.n_calls % self.freq == 0:
        if self.num_timesteps % self.freq == 0:
            self.iter += 1

            if (self.num_timesteps // self.freq) % 500 == 0 or self.iter > 3980:
                self.model.save(os.path.join(self.model_dir, f"{self.iter}.zip"))

            # if self.eval:

            #     validation_reward = self._eval_on_test(n_samples = self.n_samples)

            #     # open csv file containing everything (create it if it doesn't exist)
            #     with open(file = os.path.join(self.data_dir, "rollout_test_rewards.csv"), mode = "a+", newline = '') as csvfile:
            #         writer = csv.writer(csvfile)

            #         writer.writerow([validation_reward])

            #     # log the validation reward to tensorboard
            #     self.logger.record("rollout/ep_val_rew", validation_reward)

            #     # save best model
            #     if validation_reward > self.best_val_reward:
                    
            #         self.best_val_reward = validation_reward

            #         self.model.save(os.path.join(self.model_dir, "best_model.zip"))

        
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
        # if self.eval:
        #     with open(os.path.join(self.data_dir, "rollout_test_rewards.csv"), mode = "r", newline = "") as csvfile:
        #         reader = csv.reader(csvfile)
                
        #         # we know it has 1 value per row
        #         rollout_test_rewards = np.array([float(row[0]) for row in reader], dtype = np.float32)
            
        #     # since rollout test rewards are stored after every self.update_freq episodes
        #     episodes_axis = np.array(
        #         [
        #             self.update_freq * (iter + 1) for iter in range(rollout_test_rewards.shape[0])
        #         ], 
        #         dtype = np.float32
        #     )

        #     plt.plot(episodes_axis, rollout_test_rewards, label = "episodic reward")
        #     plt.xlabel("episodes")
        #     plt.ylabel("reward")
        #     plt.savefig(os.path.join(self.data_dir, "episodic_reward.png"))
        #     plt.close()

    def _eval_on_test(self, n_samples: int = 1) -> float:

        ep_rew = 0

        for _ in range(n_samples):

            obs, info = self.test_env.reset()

            for i in range(self.ep_len):

                action, _ = self.model.predict(obs, deterministic = True)

                obs, reward, terminated, truncated, info = self.test_env.step(action)

                ep_rew += reward
        
        ep_rew /= n_samples

        self.test_env.reset()

        return ep_rew
    

class VanillaModelManager:

    def __init__(
        self,
        algorithm: str = "PPO",
        policy_type: str = "MlpPolicy",
        env_config = {},
        env_class = SIRSGov,
        additional: dict = {},
        update_freq: int = 2,
        n_envs: int = 1,
        batch_size: int = None,
        dir: str = None,
        seed: int = 42
    ):
        
        self.algorithm = algorithm.lower()
        self.policy_type = policy_type
        
        self.env_config = env_config
        
        self.env_class = env_class
        self.train_env = self.env_class(**self.env_config)

        self.env_config = self.train_env.get_config()

        self.additional = additional

        self.update_freq = update_freq

        self.episode_length = self.env_config["T"]

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-7:].lower() != "vanilla":
            self.dir = os.path.join(self.dir, "Vanilla")

        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = int(seed)
        # print(f"seed is {self.seed}")
        # print(f"type of seed is {type(self.seed)}")

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = lambda progress: 0.0003

    def make(
        self
    ):

        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)
        
        if self.algorithm == "ppo":

            self.model = PPO(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1,
                # tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )

        elif self.algorithm == "a2c":

            self.model = A2C(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.algorithm == "dqn":

            self.model = DQN(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
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
        self,
        # save_dir
    ):
        # self.save_dir = save_dir
        self.save_dir = os.path.join(self.dir, "saved")

        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)

        if self.algorithm == "ppo":
            self.model = PPO.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "dqn":
            self.model = DQN.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        
        self.model.set_random_seed(self.seed)

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
            "learning_rate",
            "env_class",
            "episode_length",
            "n_steps",
            "log_dir",
            "save_dir",
            "config",
            "callback"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)

        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model
        self.model.save(os.path.join(self.save_dir, "model.zip"))


    def close_threads(self):

        self.model.env.close()

        # After training, flush and close TensorBoard writers
        for output_format in self.model.logger.output_formats:
            if hasattr(output_format, 'writer') and output_format.writer is not None:
                output_format.writer.flush()
                output_format.writer.close()

    def train_stop(
        self,
        N_iter: int,
        N_mu: int
    ):

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = N_mu * self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.train_env)
        )

        total_timesteps = self.n_steps * N_mu

        iter = 0

        while iter < N_iter:
            iter += 1

            self.model.learn(
                total_timesteps = total_timesteps,
                reset_num_timesteps = False,
                log_interval = 1,
                tb_log_name = "Vanilla",
                callback = self.callback
            )

        # total_timesteps = N_iter * self.n_steps * N_mu

        # self.model.learn(
        #     total_timesteps = total_timesteps, 
        #     reset_num_timesteps = False, 
        #     log_interval = 1, 
        #     tb_log_name = "Vanilla",
        #     callback = self.callback
        # )

        

        self.save()

        self.close_threads()

class RandModelManager:

    def __init__(
        self,
        algorithm: str = "PPO",
        policy_type: str = "MlpPolicy",
        env_config = {},
        env_class = RandSIRSGov,
        additional: dict = {},
        update_freq: int = 2,
        n_envs: int = 1,
        batch_size: int = None,
        dir: str = None,
        seed: int = 42
    ):
        
        self.algorithm = algorithm.lower()
        self.policy_type = policy_type
        
        self.env_config = env_config
        
        self.env_class = env_class
        self.train_env = self.env_class(**self.env_config)
        self.env_config = self.train_env.get_config()

        self.additional = additional

        self.update_freq = update_freq

        self.episode_length = self.env_config["T"]

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-4:].lower() != "rand":
            self.dir = os.path.join(self.dir, "Rand")

        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = int(seed)

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = lambda progress: 0.0003

    def make(
        self
    ):

        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)
        
        if self.algorithm == "ppo":

            self.model = PPO(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )

        elif self.algorithm == "a2c":

            self.model = A2C(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.algorithm == "dqn":

            self.model = DQN(
                policy = self.policy_type,
                env = self.env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )
        
        self.save()

    def load(
        self,
        # save_dir
    ):
        # self.save_dir = save_dir
        self.save_dir = os.path.join(self.dir, "saved")

        self.env = DummyVecEnv([lambda: self.train_env])
        self.env = VecMonitor(self.env)

        if self.algorithm == "ppo":
            self.model = PPO.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "a2c":
            self.model = A2C.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)
        elif self.algorithm == "dqn":
            self.model = DQN.load(path = os.path.join(self.save_dir, "model.zip"),  env = self.env)

        self.model.set_random_seed(self.seed)

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
            "learning_rate",
            "env_class",
            "episode_length",
            "n_steps",
            "log_dir",
            "save_dir",
            "config",
            "callback"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)

        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.model.save(os.path.join(self.save_dir, "model.zip"))

    def train_stop(
        self,
        N_iter: int,
        N_mu: int
    ):

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = N_mu * self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.train_env)
        )

        total_timesteps = self.n_steps * N_mu

        iter = 0

        while iter < N_iter:
            iter += 1

            self.model.learn(
                total_timesteps = total_timesteps,
                reset_num_timesteps = False,
                log_interval = 1,
                tb_log_name = "Rand",
                callback = self.callback
            )

        # total_timesteps = N_iter * self.n_steps * N_mu

        # self.model.learn(
        #     total_timesteps = total_timesteps, 
        #     reset_num_timesteps = False, 
        #     log_interval = 1, 
        #     tb_log_name = "Vanilla",
        #     callback = self.callback
        # )

        self.save()

class RARLModelManager:
    def __init__(
        self,
        gov_algorithm: str = "PPO",
        adv_algorithm: str = "PPO",
        gov_policy_type: str = "MlpPolicy",
        adv_policy_type: str = "MlpPolicy",
        sim_config = {},
        vanilla_class = SIRSGov,
        sim_class = SIRSSimulator,
        gov_class = RARLSIRSGov,
        adv_class = RARLSIRSAdv,
        additional: dict = {"callback_n_samples" : 3},
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

        del self.env_config["adv_action_types"]
        del self.env_config["gym_spaces_bounds"]

        self.vanilla_env = self.vanilla_class(**self.env_config)

        self.env_config = self.vanilla_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq

        self.episode_length = self.simulator.T

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-4:].lower() != "rarl":
            self.dir = os.path.join(self.dir, "RARL")
        
        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = int(seed)

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003

    
    def make(
        self
    ):

        # make gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        
        if self.gov_algorithm == "ppo":

            self.gov_model = PPO(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1,
                # tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )

        elif self.gov_algorithm == "dqn":


            self.gov_model = DQN(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.gov_algorithm == "a2c":

            self.gov_model = A2C(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
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
                verbose = 1,
                # tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )
        elif self.adv_algorithm == "dqn":


            self.adv_model = DQN(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.adv_algorithm == "a2c":

            self.adv_model = A2C(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

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


        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.save()

    def load(
        self,
        # save_dir: str
    ):
        # self.save_dir = save_dir
        self.save_dir = os.path.join(self.dir, "saved")

        with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
            self.config = json.load(file)

        # load gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        

        if self.gov_algorithm == "ppo":
            self.gov_model = PPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "dqn":
            self.gov_model = DQN.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_random_seed(self.seed)
        self.adv_model.set_random_seed(self.seed)

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
            "save_dir",
            "config",
            "env_config"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)


        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.gov_model.save(os.path.join(self.save_dir, "gov_model.zip"))

        self.adv_model.save(os.path.join(self.save_dir, "adv_model.zip"))

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

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = N_mu * self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.vanilla_env)
        )

        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = self.n_steps * N_nu

        iter = 0
        while iter < N_iter:
            iter += 1

            self.gov_model.learn(
                total_timesteps = gov_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "RARL", 
                callback = self.callback
            )

            self.adv_model.learn(
                total_timesteps = adv_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "Adv"
            )

        
        self.save()

        self.close_threads()


class RARLModelManager2:
    def __init__(
        self,
        gov_algorithm: str = "PPO",
        adv_algorithm: str = "PPO",
        gov_policy_type: str = "MlpPolicy",
        adv_policy_type: str = "MlpPolicy",
        sim_config = {},
        vanilla_class = SIRSGov,
        sim_class = SIRSSimulator2,
        gov_class = RARLSIRSGov2,
        adv_class = RARLSIRSAdv2,
        additional: dict = {},
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

        del self.env_config["adv_action_types"]
        del self.env_config["gym_spaces_bounds"]

        self.vanilla_env = self.vanilla_class(**self.env_config)

        self.env_config = self.vanilla_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq

        self.episode_length = self.simulator.T

        # number of steps before update formula
        self.n_steps = self.update_freq * self.episode_length

        self.n_envs = n_envs

        if dir == None:
            raise ValueError("dir cannot be None; needs to be passsed")
        self.dir = deepcopy(dir)
        if self.dir[-5:].lower() != "rarl2":
            self.dir = os.path.join(self.dir, "RARL2")
        
        if batch_size == None:
            # self.batch_size = self.n_steps
            self.batch_size = compute_batch_size(self.n_steps)
        else:
            self.batch_size = batch_size
        
        self.seed = int(seed)

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003

    
    def make(
        self
    ):

        # make gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        
        if self.gov_algorithm == "ppo":

            self.gov_model = PPO(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )

        elif self.gov_algorithm == "dqn":


            self.gov_model = DQN(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.gov_algorithm == "a2c":

            self.gov_model = A2C(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
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
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )
        elif self.adv_algorithm == "dqn":


            self.adv_model = DQN(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                train_freq = int(self.n_steps / self.episode_length),
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.adv_algorithm == "a2c":

            self.adv_model = A2C(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = int(self.n_steps / self.episode_length),
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )


        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.save()

    def load(
        self,
        # save_dir: str
    ):
        self.save_dir = os.path.join(self.dir, "saved")
        # self.save_dir = save_dir

        with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
            self.config = json.load(file)

        # load gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        

        if self.gov_algorithm == "ppo":
            self.gov_model = PPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "dqn":
            self.gov_model = DQN.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_random_seed(self.seed)
        self.adv_model.set_random_seed(self.seed)

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
            "save_dir",
            "config",
            "env_config"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)


        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.gov_model.save(os.path.join(self.save_dir, "gov_model.zip"))

        self.adv_model.save(os.path.join(self.save_dir, "adv_model.zip"))


    def train_stop(
        self,
        N_iter,
        N_mu,
        N_nu: int = None
    ):

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = N_mu * self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.vanilla_env)
        )

        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = int(N_nu * self.n_steps / self.episode_length)

        iter = 0
        while iter < N_iter:
            iter += 1

            self.gov_model.learn(
                total_timesteps = gov_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "RARL2", 
                callback = self.callback
            )

            self.adv_model.learn(
                total_timesteps = adv_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "Adv"
            )

        self.save()

class ARLModelManager:
    def __init__(
        self,
        gov_algorithm: str = "PPO",
        adv_algorithm: str = "PPO",
        gov_policy_type: str = "MlpPolicy",
        adv_policy_type: str = "MlpPolicy",
        sim_config = {},
        vanilla_class = SIRSGov,
        sim_class = SIRSSimulator2,
        gov_class = RARLSIRSGov2,
        adv_class = RARLSIRSAdv2,
        additional: dict = {},
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

        del self.env_config["adv_action_types"]
        del self.env_config["gym_spaces_bounds"]

        self.vanilla_env = self.vanilla_class(**self.env_config)

        self.env_config = self.vanilla_env.get_config()
        
        self.additional = additional
        if "callback_n_samples" not in self.additional:
            self.additional["callback_n_samples"] = 1

        self.update_freq = update_freq

        self.episode_length = self.simulator.T

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
        
        self.seed = int(seed)

        self.log_dir = os.path.join(self.dir, "logs")

        self.learning_rate = 0.0003

    
    def make(
        self
    ):

        # make gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        
        if self.gov_algorithm == "ppo":

            self.gov_model = PPO(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                batch_size = self.batch_size,
                clip_range = 0.2,
                verbose = 1,
                # tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )

        elif self.gov_algorithm == "dqn":


            self.gov_model = DQN(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                train_freq = self.n_steps,
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.gov_algorithm == "a2c":

            self.gov_model = A2C(
                policy = self.gov_policy_type,
                env = self.gov_env,
                learning_rate = self.learning_rate,
                n_steps = self.n_steps,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        if self.adv_algorithm == "ppo":
            
            self.adv_model = PPO(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = int(self.n_steps / self.episode_length),
                # batch_size = self.batch_size,
                batch_size = 2,
                clip_range = 0.2,
                verbose = 1,
                # tensorboard_log = self.log_dir,
                seed = self.seed,
                ent_coef=0.01
            )
        elif self.adv_algorithm == "dqn":


            self.adv_model = DQN(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                train_freq = int(self.n_steps / self.episode_length),
                batch_size = self.batch_size,
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

        elif self.adv_algorithm == "a2c":

            self.adv_model = A2C(
                policy = self.adv_policy_type,
                env = self.adv_env,
                learning_rate = self.learning_rate,
                n_steps = int(self.n_steps / self.episode_length),
                verbose = 1,
                tensorboard_log = self.log_dir,
                seed = self.seed
            )

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


        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.save()

    def load(
        self,
        # save_dir: str
    ):
        self.save_dir = os.path.join(self.dir, "saved")
        # self.save_dir = save_dir

        with open(os.path.join(self.save_dir, "model_config.json"), "r") as file:
            self.config = json.load(file)

        # load gov_env and adv_env
        self.gov_env = DummyVecEnv([lambda: self.gov_train_env])
        self.gov_env = VecMonitor(self.gov_env)

        self.adv_env = DummyVecEnv([lambda: self.adv_train_env])
        self.adv_env = VecMonitor(self.adv_env)
        

        if self.gov_algorithm == "ppo":
            self.gov_model = PPO.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "a2c":
            self.gov_model = A2C.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)
        elif self.gov_algorithm == "dqn":
            self.gov_model = DQN.load(path = os.path.join(self.save_dir, "gov_model.zip"), env = self.gov_env)
            self.adv_model = PPO.load(path = os.path.join(self.save_dir, "adv_model.zip"), env = self.adv_env)

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
        

        self.gov_train_env.set_adv_agent(self.adv_model)
        self.adv_train_env.set_gov_agent(self.gov_model)

        self.gov_model.set_random_seed(self.seed)
        self.adv_model.set_random_seed(self.seed)

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
            "save_dir",
            "config",
            "env_config"
        }

        self.config = {}

        for key, value in vars(self).items():
            if key not in keys_to_remove:
                self.config[key] = deepcopy(value)


        # save model config dict
        with open(os.path.join(self.save_dir, "model_config.json"), 'w') as file:
            json.dump(self.config, file, indent = 4)

        # save sb3 model and env
        self.gov_model.save(os.path.join(self.save_dir, "gov_model.zip"))

        self.adv_model.save(os.path.join(self.save_dir, "adv_model.zip"))


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

        self.callback = CustomCallback(
            ep_len = self.episode_length,
            update_freq = N_mu * self.update_freq,
            save_path = self.dir,
            verbose = 1,
            test_env = deepcopy(self.vanilla_env)
        )

        gov_total_timesteps = self.n_steps * N_mu
        if N_nu == None:
            N_nu = N_mu
        adv_total_timesteps = int(N_nu * self.n_steps / self.episode_length)

        iter = 0
        while iter < N_iter:
            iter += 1

            self.gov_model.learn(
                total_timesteps = gov_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "ARL", 
                callback = self.callback
            )

            self.adv_model.learn(
                total_timesteps = adv_total_timesteps, 
                reset_num_timesteps = False, 
                log_interval = 1, 
                tb_log_name = "Adv"
            )

        self.save()

        self.close_threads()


           
class ModelTester:

    def __init__(
        self,
        algo: str,
        dir: str,
        test_env,
        seed: int = 42,
        norm: bool = False
    ):
        algo = algo.lower()
        allowed_algos = ["ppo", "a2c", "dqn"]
        if not algo in allowed_algos:
            raise ValueError(f"algo must be in {str(allowed_algos)}")
            
        self.algo = algo

        self.norm = False
        
        if not (dir[-4:].lower() == "rarl" or dir[-7:].lower() == "vanilla" or dir[-4:].lower() == "rand" or dir[-5:].lower() == "rarl2" or dir[-3:].lower() == "arl"):
            raise ValueError("vanilla or rarl or rand or rarl2 or arl directory needs to be passed")
        self.dir = dir

        self.model_dir = os.path.join(self.dir, "models")
        self.save_dir = os.path.join(self.dir, "saved")

        self.test_env = test_env

        self.episode_length = self.test_env.T

        self.seed = int(seed)


    def load(
        self,
        checkpoint_no: int
    ):
        
        if not os.path.exists(os.path.join(self.model_dir, f"{checkpoint_no}.zip")):
            
            path_checked = os.path.join(self.model_dir, f"{checkpoint_no}.zip")
            error_details = f"model_dir: {self.model_dir} \n checkpoint_no: {checkpoint_no} \n path checked: {path_checked}"
            raise ValueError(f"model checkpoint doesn't exist\n{error_details}")
        
        # load env

        self.env = DummyVecEnv([lambda: self.test_env])

        # load model

        if self.algo == "ppo":
            
            self.model = PPO.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
           
        elif self.algo == "dqn":
            self.model = DQN.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )
        elif self.algo == "a2c":
            self.model = A2C.load(
                path = os.path.join(self.model_dir, f"{checkpoint_no}.zip"),
                env = self.env
            )

        self.model.set_random_seed(self.seed)

    def load_last(
        self
    ):
        is_vanilla = True
        if self.dir[-4:].lower() == "rarl" or self.dir[-3:].lower() == "arl":
            is_vanilla = False

        self.env = DummyVecEnv([lambda: deepcopy(self.test_env)])

        if is_vanilla:

            model_path = os.path.join(self.save_dir, "model.zip")
            model_path =os.path.join(self.model_dir, "4000.zip")
            if not os.path.exists(model_path):
                raise ValueError(f"model doesn't exist in {model_path}")

            if self.norm:
                self.env = VecNormalize.load(
                    load_path = os.path.join(self.save_dir, "env.pkl"),
                    venv = self.env
                )
        
        else:

            model_path = os.path.join(self.save_dir, "gov_model.zip")
            model_path =os.path.join(self.model_dir, "4000.zip")
            if not os.path.exists(model_path):
                raise ValueError(f"model doesn't exist in {model_path}")

            if self.norm:
                self.env = VecNormalize.load(
                    load_path = os.path.join(self.save_dir, "gov_env.pkl"),
                    venv = self.env
                )
            
        
        if self.algo == "ppo":
            self.model = PPO.load(
                path = model_path
            )

        self.model.set_random_seed(self.seed)


    def compute_reward(
        self,
        n_samples: int = 1
    ):
        
        ep_rew_mean = 0.0

        obs, info = self.test_env.reset()

        for j in range(self.episode_length):

            action, _ = self.model.predict(obs, deterministic = True)

            obs, reward, term, trun, info = self.test_env.step(action)

            ep_rew_mean += reward

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

                adv_action, _ = adv_model.predict(obs, deterministic = True)
                self.test_env.take_adv_actions(
                    adv_action,
                    adv_action_types
                )
                
                # # taking adversarial action
                # action_np = np.array(adv_action, dtype = np.float32)
                # for k in range(action_np.shape[0]):
                    
                #     self.test_env.model.params[
                #         jl.Symbol(self.test_env.actions_to_properties[adv_action_types[k]])
                #     ] = adv_action[k]

                # if self.norm:
                #     norm_obs = self.env.normalize_obs(obs)
                # else:
                #     norm_obs = obs

                action, _ = self.model.predict(obs, deterministic = True)

                obs, reward, term, trun, info = self.test_env.step(action)

                ep_rew_mean += reward

        ep_rew_mean /= n_samples

        self.test_env.reset()

        return ep_rew_mean

        