from VanillaGov import VanillaGovABC
from model import VanillaModelManager, RARLModelManager, ModelTester
from copy import deepcopy
import os
import json
import numpy as np
import dill as picklefrom 
from model import VanillaModelManager, RARLModelManager, ARLModelManager, ModelTester
from copy import deepcopy
import os
import json
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import scienceplots
from stable_baselines3.common.vec_env import VecNormalize

class EvaluateRobustness:

    # TODO: change some of these to make more legible
    param_to_label = {
        "consumption_wealth_ratio" : "Ratio of Consumption to Wealth",
        "firm_invest_prob" : "Probability of Firm Investment",
        "memory_parameter" : "Memory Parameter",
        "labour_prod" : "Productivity of Labour",
        "capital_prod" : "Productivity of Capital"
    }

    param_to_name = {
        "cwr" : "Ratio of Consumption to Wealth",
        "fip" : "Probability of Firm Investment",
        "mp" : "Memory Parameter",
        "lp" : "Productivity of Labour",
        "cp" : "Productivity of Capital"
    }

    def __init__(
        self,
        dir: str = None,
        n_seeds: int = 10,
        n_train_seeds: int = 3
    ):
        if dir == None:
            raise ValueError("dir cannot be None")
        self.dir = dir

        self.n_seeds = n_seeds
        self.seeds = np.array([i for i in range(1, 1 + self.n_seeds)], dtype = np.int64)

        self.n_train_seeds = n_train_seeds
        self.train_seeds = np.array([i for i in range(42, 42 + self.n_train_seeds)], dtype = np.int64)
        
        self.param_datasets = {}
        self.param_shifted_envs = {}

        # make vanilla and rarl and arl models
        # since we surely have a vanilla model
        self._make_vanilla_models()
        self._make_arl_models()
        self._make_rarl_models()
        
        self.sim_config = deepcopy(self.arl_models[0].sim_config)

        self.env_config = deepcopy(self.vanilla_models[0].env_config)
        self.env_class = deepcopy(self.vanilla_models[0].env_class)

        self.plot_dir = os.path.join(self.dir, "robustness_comparison_plots")
        os.makedirs(self.plot_dir, exist_ok = True)

        self.adv_models = {}

        self.adv_datasets = {}
        self.adv_shifted_envs = {}

    def add_param_shift_one(
        self,
        name,
        shift
    ):
        """
        plots parameter shift performance 

        args:
            shift, which contains
                shifted_param
                range: lower and upper bound of parameter shift
                n_values: number of values in the range of values
                logspace: bool (false by default) -- whether the values should be generated in logspace
            name
        """

        logspace = False
        if "logspace" in shift:
            logspace = shift["logspace"]

        low, high = shift["range"]
        n_values = shift["n_values"]
        param = shift["param"]

        if not logspace:
            values = np.linspace(low, high, n_values, dtype = np.float32)
        else:
            values = np.logspace(np.log10(low), np.log10(high), n_values, dtype = np.float32)

        if "values" in shift:
            values = shift["values"]
            if not isinstance(values, np.ndarray):
                values = np.array(values)

        self.param_shifted_envs[name] = (param, values)

        self._add_datasets_param_shift_one(name)

        self._plot_for_param_shift_one(name)

    
    def _add_datasets_param_shift_one(
        self,
        name
    ):
        # print(f"adding datasets for {name}")

        shifted_env_config = deepcopy(self.env_config)
        shifted_env_config["additional"]["change"] = True

        param, values = self.param_shifted_envs[name]

        n_values = values.shape[0]

        vanilla_dataset = np.zeros(shape = (self.n_train_seeds, n_values, self.n_seeds), dtype = np.float32)
        arl_dataset = np.zeros(shape = (self.n_train_seeds, n_values, self.n_seeds), dtype = np.float32)
        rarl_dataset = np.zeros(shape = (self.n_train_seeds, n_values, self.n_seeds), dtype = np.float32)

        
        for train_seed_iter in range(self.n_train_seeds):
            
            train_seed = self.train_seeds[train_seed_iter]

            # print(f"computing datasets for train seed {train_seed}")

            curr_dir = os.path.join(self.dir, str(train_seed))

            for value_iter in range(n_values):
                value = values[value_iter]

                shifted_env_config["additional"]["params_to_change"] = {
                    param: value
                }

                for seed_iter in range(self.n_seeds):
                    seed = self.seeds[seed_iter]

                    shifted_env_config["seed"] = seed

                    shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                    vanilla_model_tester = ModelTester(
                        # algo = "PPO",
                        algo = self.env_config["algorithm"],
                        dir = os.path.join(curr_dir, "Vanilla"),
                        test_env = shifted_env_v,
                        norm = True,
                    )
                    vanilla_model_tester.load_last()

                    vanilla_dataset[train_seed_iter][value_iter][seed_iter] = vanilla_model_tester.compute_reward(n_samples = 1)

                    arl_model_tester = ModelTester(
                        # algo = "PPO",
                        algo = self.env_config["algorithm"],
                        dir = os.path.join(curr_dir, "ARL"),
                        test_env = shifted_env_a,
                        norm = True,
                    )
                    arl_model_tester.load_last()

                    arl_dataset[train_seed_iter][value_iter][seed_iter] = arl_model_tester.compute_reward(n_samples = 1)

                    rarl_model_tester = ModelTester(
                        # algo = "PPO",
                        algo = self.env_config["algorithm"],
                        dir = os.path.join(curr_dir, "RARL"),
                        test_env = shifted_env_r,
                        norm = True,
                    )
                    rarl_model_tester.load_last()

                    rarl_dataset[train_seed_iter][value_iter][seed_iter] = rarl_model_tester.compute_reward(n_samples = 1)

        self.param_datasets[f"vanilla_{name}"] = vanilla_dataset
        self.param_datasets[f"arl_{name}"] = arl_dataset
        self.param_datasets[f"rarl_{name}"] = rarl_dataset


        
        self.param_datasets_dir = os.path.join(self.dir, f"datasets")
        os.makedirs(self.param_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.param_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "w") as file:
            json.dump(vanilla_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "w") as file:
            json.dump(arl_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "w") as file:
            json.dump(rarl_dataset.tolist(), file)

        

    def _plot_for_param_shift_one(
        self,
        name
    ):
        """
        plots the following graphs, given that this is a parameter shift:
            for each train_seed, a comparison, on the range of values, of vanilla vs arl vs rarl -- in separate graphs for each seed
                this should have the std error of mean bars in on each data point
            one plot which compares the three models when taking an average over train seeds
            
        """

        # plt.style.use(['science', 'no-latex'])
        plt.style.use('science')


        # plot_dir = os.path.join(self.plot_dir, name)
        # os.makedirs(plot_dir, exist_ok = True)

        param, values = self.param_shifted_envs[name]

        vanilla_dataset = self.param_datasets[f"vanilla_{name}"]
        arl_dataset = self.param_datasets[f"arl_{name}"]
        rarl_dataset = self.param_datasets[f"rarl_{name}"]

        # for train_seed_iter in range(self.n_train_seeds):
        #     # plot the arl vs rarl vs vanilla comparison graphs 

        #     train_seed = self.train_seeds[train_seed_iter]

        #     this_vanilla_dataset = vanilla_dataset[train_seed_iter]
        #     this_arl_dataset = arl_dataset[train_seed_iter]
        #     this_rarl_dataset = rarl_dataset[train_seed_iter]

        #     vanilla_means = np.mean(this_vanilla_dataset, axis = 1)
        #     arl_means = np.mean(this_arl_dataset, axis = 1)
        #     rarl_means = np.mean(this_rarl_dataset, axis = 1)

        #     vanilla_std = np.std(this_vanilla_dataset, axis = 1)
        #     arl_std = np.std(this_arl_dataset, axis = 1)
        #     rarl_std = np.std(this_rarl_dataset, axis = 1)

        #     # turning plotting off
        #     plt.ioff()

        #     plt.errorbar(values, vanilla_means, yerr = vanilla_std, color = "green", ecolor = "green", fmt = "D", capsize = 5)
        #     plt.plot(values, vanilla_means, color = "green", label = "Vanilla", alpha = 0.5)

        #     plt.errorbar(values, arl_means, yerr = arl_std, color = "lightblue", ecolor = "lightblue", fmt = "X", capsize = 5)
        #     plt.plot(values, arl_means, color = "lightblue", label = "Episode-Adv", alpha = 0.5)

        #     plt.errorbar(values, rarl_means, yerr = rarl_std, color = "darkblue", ecolor = "darkblue", fmt = "o", capsize = 5)
        #     plt.plot(values, rarl_means, color = "darkblue", label = "Step-Adv", alpha = 0.5)

        #     plt.xlabel(self.param_to_label[param])
        #     plt.ylabel("Rewards")

        #     plt.title(f"{self.param_to_label[param]} from {values[0]:.2f} to {values[-1]:.2f}")

        #     plt.legend()

        #     plt.tight_layout()

        #     plt.savefig(f"{plot_dir}/TS_{train_seed}.png", dpi = 300)

        #     plt.close()
        
        # # FOR EACH VALUE, FOR EACH MODEL TYPE, TAKING THE AVERAGE OF n_train_seeds MODELS AND PLOTTING WITH STD DEVS (OVER n_seeds TEST SEEDS)

        # avg_vanilla_dataset = np.mean(vanilla_dataset, axis = 0)
        # avg_arl_dataset = np.mean(arl_dataset, axis = 0)
        # avg_rarl_dataset = np.mean(rarl_dataset, axis = 0)

        # avg_vanilla_means = np.mean(avg_vanilla_dataset, axis = 1)
        # avg_arl_means = np.mean(avg_arl_dataset, axis = 1)
        # avg_rarl_means = np.mean(avg_rarl_dataset, axis = 1)

        # avg_vanilla_std = np.std(avg_vanilla_dataset, axis = 1)
        # avg_arl_std = np.std(avg_arl_dataset, axis = 1)
        # avg_rarl_std = np.std(avg_rarl_dataset, axis = 1)

        # plt.ioff()

        # plt.errorbar(values, avg_vanilla_means, yerr = avg_vanilla_std, color = "green", ecolor = "green", fmt = "D", capsize = 5)
        # plt.plot(values, avg_vanilla_means, color = "green", label = "Vanilla", alpha = 0.5)

        # plt.errorbar(values, avg_arl_means, yerr = avg_arl_std, color = "lightblue", ecolor = "lightblue", fmt = "X", capsize = 5)
        # plt.plot(values, avg_arl_means, color = "lightblue", label = "Episode-Adv", alpha = 0.5)

        # plt.errorbar(values, avg_rarl_means, yerr = avg_rarl_std, color = "darkblue", ecolor = "darkblue", fmt = "o", capsize = 5)
        # plt.plot(values, avg_rarl_means, color = "darkblue", label = "Step-Adv", alpha = 0.5)

        # plt.xlabel(self.param_to_label[param])
        # plt.ylabel("Rewards")

        # plt.title(f"{self.param_to_label[param]} from {values[0]:.2f} to {values[-1]:.2f}")

        # plt.legend()

        # plt.tight_layout()

        # plt.savefig(f"{plot_dir}/average_over_seeds.png", dpi = 300)

        # plt.close()

        


        # # PLOTTING WITH FILL BETWEEN

        # vanilla_seed_means = np.mean(vanilla_dataset, axis = 2)
        # arl_seed_means = np.mean(arl_dataset, axis = 2)
        # rarl_seed_means = np.mean(rarl_dataset, axis = 2)

        # vanilla_means = np.mean(vanilla_seed_means, axis = 0)
        # arl_means = np.mean(arl_seed_means, axis = 0)
        # rarl_means = np.mean(rarl_seed_means, axis = 0)

        # vanilla_min = np.min(vanilla_seed_means, axis = 0)
        # arl_min = np.min(arl_seed_means, axis = 0)
        # rarl_min = np.min(rarl_seed_means, axis = 0)

        # vanilla_max = np.max(vanilla_seed_means, axis = 0)
        # arl_max = np.max(arl_seed_means, axis = 0)
        # rarl_max = np.max(rarl_seed_means, axis = 0)

        # # vanilla vs arl

        # plt.ioff()

        # plt.plot(values, vanilla_means, color = "green", label = "Vanilla")
        # plt.fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        # plt.plot(values, arl_means, color = "blue", label = "Episode-Adv")
        # plt.fill_between(values, arl_min, arl_max, color = "blue", alpha = 0.5)

        # plt.xlabel(self.param_to_label[param])
        # plt.ylabel("Rewards")

        # # plt.title(f"{self.param_to_label[param]} from {values[0]:.2f} to {values[-1]:.2f}")

        # plt.legend()

        # plt.tight_layout()

        # plt.savefig(f"{plot_dir}/vanilla_vs_arl.png", dpi = 300)

        # plt.close()

        # # vanilla vs rarl

        # plt.ioff()

        # plt.plot(values, vanilla_means, color = "green", label = "Vanilla")
        # plt.fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        # plt.plot(values, rarl_means, color = "blue", label = "Step-Adv")
        # plt.fill_between(values, rarl_min, rarl_max, color = "blue", alpha = 0.5)

        # plt.xlabel(self.param_to_label[param])
        # plt.ylabel("Rewards")

        # # plt.title(f"{self.param_to_label[param]} from {values[0]:.2f} to {values[-1]:.2f}")

        # plt.legend()

        # plt.tight_layout()

        # plt.savefig(f"{plot_dir}/vanilla_vs_rarl.png", dpi = 300)

        # plt.close()

        # making both figures in one

        # PLOTTING WITH FILL BETWEEN

        vanilla_seed_means = np.mean(vanilla_dataset, axis = 2)
        arl_seed_means = np.mean(arl_dataset, axis = 2)
        rarl_seed_means = np.mean(rarl_dataset, axis = 2)

        vanilla_means = np.mean(vanilla_seed_means, axis = 0)
        arl_means = np.mean(arl_seed_means, axis = 0)
        rarl_means = np.mean(rarl_seed_means, axis = 0)

        vanilla_min = np.min(vanilla_seed_means, axis = 0)
        arl_min = np.min(arl_seed_means, axis = 0)
        rarl_min = np.min(rarl_seed_means, axis = 0)

        vanilla_max = np.max(vanilla_seed_means, axis = 0)
        arl_max = np.max(arl_seed_means, axis = 0)
        rarl_max = np.max(rarl_seed_means, axis = 0)


        # Create the 1x3 subplot layout
        # fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        # fig, axs = plt.subplots(1, 2, figsize=(6, 2))

        # plt.ioff()

        # # plt.rcParams.update({'font.size': 8})  # Set the global font size to 12

        # # plt.rcParams.update({
        # #     'axes.labelsize': 8,    # Font size for x and y labels
        # #     'xtick.labelsize': 8,   # Font size for x-axis tick labels
        # #     'ytick.labelsize': 8,   # Font size for y-axis tick labels
        # #     'axes.titlesize': 8     # Font size for the subplot titles
        # # })

        # axs[0].plot(values, vanilla_means, color = "green", label = "Vanilla")
        # axs[0].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        # axs[0].plot(values, arl_means, color = "blue", label = "Episode-Adv")
        # axs[0].fill_between(values, arl_min, arl_max, color = "blue", alpha = 0.5)

        # axs[0].set_xlabel(self.param_to_label[param])
        # axs[0].set_ylabel("Rewards")

        # axs[0].margins(0.2)  # Increase margins

        # axs[0].legend()

        # y_lim_min, y_lim_max = axs[0].get_ylim()

        # y_min = np.minimum(vanilla_min.min(), arl_min.min())
        # y_max = np.maximum(vanilla_max.max(), arl_max.max())

        # axs[0].set_ylim(
        #     [
        #         y_lim_min,
        #         y_lim_max + 0.5 * (y_max - y_min)
        #     ]
        # )


        # axs[1].plot(values, vanilla_means, color = "green", label = "Vanilla")
        # axs[1].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        # axs[1].plot(values, rarl_means, color = "blue", label = "Step-Adv")
        # axs[1].fill_between(values, rarl_min, rarl_max, color = "blue", alpha = 0.5)

        # axs[1].set_xlabel(self.param_to_label[param])
        # axs[1].set_ylabel("Rewards")

        # axs[1].margins(0.2)  # Increase margins

        # axs[1].legend()

        # y_lim_min, y_lim_max = axs[1].get_ylim()

        # y_min = np.minimum(vanilla_min.min(), rarl_min.min())
        # y_max = np.maximum(vanilla_max.max(), rarl_max.max())

        # axs[1].set_ylim(
        #     [
        #         y_lim_min,
        #         y_lim_max + 0.5 * (y_max - y_min)
        #     ]
        # )

        # fig.suptitle(self.param_to_label[param])

        # plt.tight_layout()

        # # plt.savefig(f"{plot_dir}/vanilla_vs_rarl_vs_arl.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/science-plots/fixed-one-2", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/science-plots/fixed-one-2/{name}.png", dpi = 300)



        plt.rcParams.update({'font.size': 12})  # Set the global font size to 12

        plt.rcParams.update({
            'axes.labelsize': 14,    # Font size for x and y labels
            'xtick.labelsize': 16,   # Font size for x-axis tick labels
            'ytick.labelsize': 16,   # Font size for y-axis tick labels
            'axes.titlesize': 12     # Font size for the subplot titles
        })

        fig, axs = plt.subplots(1, 2, figsize=(9, 3))

        axs[0].plot(values, vanilla_means, color = "green", label = "Vanilla")
        axs[0].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[0].plot(values, arl_means, color = "blue", label = "Episode-Adv")
        axs[0].fill_between(values, arl_min, arl_max, color = "blue", alpha = 0.5)

        axs[0].set_xlabel(self.param_to_label[param])
        axs[0].set_ylabel("Rewards")

        axs[0].margins(0.2)  # Increase margins

        axs[0].legend()

        y_lim_min, y_lim_max = axs[0].get_ylim()

        y_min = np.minimum(vanilla_min.min(), arl_min.min())
        y_max = np.maximum(vanilla_max.max(), arl_max.max())

        axs[0].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )


        axs[1].plot(values, vanilla_means, color = "green", label = "Vanilla")
        axs[1].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[1].plot(values, rarl_means, color = "blue", label = "Step-Adv")
        axs[1].fill_between(values, rarl_min, rarl_max, color = "blue", alpha = 0.5)

        axs[1].set_xlabel(self.param_to_label[param])
        # axs[1].set_ylabel("Rewards")

        axs[1].margins(0.2)  # Increase margins

        axs[1].legend()

        y_lim_min, y_lim_max = axs[1].get_ylim()

        y_min = np.minimum(vanilla_min.min(), rarl_min.min())
        y_max = np.maximum(vanilla_max.max(), rarl_max.max())

        axs[1].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )

        # fig.set_title(self.param_to_label[param])

        plt.tight_layout()

        # plt.savefig(f"{plot_dir}/vanilla_vs_rarl_vs_arl.png", dpi = 300)
        # plt.savefig(f"{plot_dir}/{name}.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/science-plots/fixed-one-2", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/science-plots/fixed-one-2/{name}.png", dpi = 300)

        os.makedirs(f"{self.plot_dir}/fixed-one", exist_ok=True)
        plt.savefig(f"{self.plot_dir}/fixed-one/{name}.png", dpi = 300)

        plt.close()


        

    def plot_existing_datasets_one_shift(
        self,
        name,
        shift
    ):
        
        # plt.style.use(['science', 'no-latex'])
        plt.style.use('science')
        
        logspace = False
        if "logspace" in shift:
            logspace = shift["logspace"]

        low, high = shift["range"]
        n_values = shift["n_values"]
        param = shift["param"]

        if not logspace:
            values = np.linspace(low, high, n_values, dtype = np.float32)
        else:
            values = np.logspace(np.log10(low), np.log10(high), n_values, dtype = np.float32)

        if "values" in shift:
            values = shift["values"]
            if not isinstance(values, np.ndarray):
                values = np.array(values)

        self.param_shifted_envs[name] = (param, values)
    
        self.param_datasets_dir = os.path.join(self.dir, f"datasets")
        os.makedirs(self.param_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.param_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "r") as file:
            vanilla_dataset = np.array(json.load(file))

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "r") as file:
            arl_dataset = np.array(json.load(file))

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "r") as file:
            rarl_dataset = np.array(json.load(file))

        plot_dir = os.path.join(self.plot_dir, "fixed-one")
        os.makedirs(plot_dir, exist_ok = True)

        param, values = self.param_shifted_envs[name]

        # vanilla_dataset = self.param_datasets[f"vanilla_{name}"]
        # arl_dataset = self.param_datasets[f"arl_{name}"]
        # rarl_dataset = self.param_datasets[f"rarl_{name}"]
        
        # PLOTTING WITH FILL BETWEEN

        vanilla_seed_means = np.mean(vanilla_dataset, axis = 2)
        arl_seed_means = np.mean(arl_dataset, axis = 2)
        rarl_seed_means = np.mean(rarl_dataset, axis = 2)

        vanilla_means = np.mean(vanilla_seed_means, axis = 0)
        arl_means = np.mean(arl_seed_means, axis = 0)
        rarl_means = np.mean(rarl_seed_means, axis = 0)

        vanilla_min = np.min(vanilla_seed_means, axis = 0)
        arl_min = np.min(arl_seed_means, axis = 0)
        rarl_min = np.min(rarl_seed_means, axis = 0)

        vanilla_max = np.max(vanilla_seed_means, axis = 0)
        arl_max = np.max(arl_seed_means, axis = 0)
        rarl_max = np.max(rarl_seed_means, axis = 0)

        plt.ioff()
        # Create the 1x3 subplot layout
        # fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        

        plt.rcParams.update({'font.size': 12})  # Set the global font size to 12

        plt.rcParams.update({
            'axes.labelsize': 14,    # Font size for x and y labels
            'xtick.labelsize': 16,   # Font size for x-axis tick labels
            'ytick.labelsize': 16,   # Font size for y-axis tick labels
            'axes.titlesize': 12     # Font size for the subplot titles
        })

        fig, axs = plt.subplots(1, 2, figsize=(9, 3))

        axs[0].plot(values, vanilla_means, color = "green", label = "Vanilla")
        axs[0].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[0].plot(values, arl_means, color = "blue", label = "Episode-Adv")
        axs[0].fill_between(values, arl_min, arl_max, color = "blue", alpha = 0.5)

        axs[0].set_xlabel(self.param_to_label[param])
        axs[0].set_ylabel("Rewards")

        axs[0].margins(0.2)  # Increase margins

        axs[0].legend()

        y_lim_min, y_lim_max = axs[0].get_ylim()

        y_min = np.minimum(vanilla_min.min(), arl_min.min())
        y_max = np.maximum(vanilla_max.max(), arl_max.max())

        axs[0].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )


        axs[1].plot(values, vanilla_means, color = "green", label = "Vanilla")
        axs[1].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[1].plot(values, rarl_means, color = "blue", label = "Step-Adv")
        axs[1].fill_between(values, rarl_min, rarl_max, color = "blue", alpha = 0.5)

        axs[1].set_xlabel(self.param_to_label[param])
        # axs[1].set_ylabel("Rewards")

        axs[1].margins(0.2)  # Increase margins

        axs[1].legend()

        y_lim_min, y_lim_max = axs[1].get_ylim()

        y_min = np.minimum(vanilla_min.min(), rarl_min.min())
        y_max = np.maximum(vanilla_max.max(), rarl_max.max())

        axs[1].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )

        # fig.set_title(self.param_to_label[param])

        plt.tight_layout()

        # plt.savefig(f"{plot_dir}/vanilla_vs_rarl_vs_arl.png", dpi = 300)
        # plt.savefig(f"{plot_dir}/{name}.png", dpi = 300)



        os.makedirs(f"{self.plot_dir}/science-plots/fixed-one", exist_ok=True)
        plt.savefig(f"{self.plot_dir}/science-plots/fixed-one/{name}.png", dpi = 300)

        plt.close()




    def add_param_shift_two(
        self,
        name,
        shifts,
    ):
        """
        plots parameter shift performance 

        args:
            shifts, an array which contains 2 dicts, each of which contains:
                shifted_param: a list of size 2 
                range: lower and upper bound of parameter shift
                n_values: number of values in the range of values
                logspace: bool (false by default) -- whether the values should be generated in logspace
            name
        """

        self.param_shifted_envs[name] = []

        for shift in shifts:

            logspace = False
            if "logspace" in shift:
                logspace = shift["logspace"]

            low, high = shift["range"]
            n_values = shift["n_values"]
            param = shift["param"]

            if not logspace:
                values = np.linspace(low, high, n_values, dtype = np.float32)
            else:
                values = np.logspace(np.log10(low), np.log10(high), n_values, dtype = np.float32)

            if "values" in shift:
                values = shift["values"]
                if not isinstance(values, np.ndarray):
                    values = np.array(values)

            self.param_shifted_envs[name].append((param, values))

        self._add_datasets_param_shift_two(name)

        self._plot_for_param_shift_two(name)

    
    def _add_datasets_param_shift_two(
        self,
        name
    ):
        param_1, values_1 = self.param_shifted_envs[name][0]
        param_2, values_2 = self.param_shifted_envs[name][1]

        n_values_1 = values_1.shape[0]
        n_values_2 = values_2.shape[0]

        n_seeds = np.minimum(self.n_seeds, 3)

        vanilla_dataset = np.zeros(shape = (self.n_train_seeds, n_values_1, n_values_2, n_seeds), dtype = np.float32)
        arl_dataset = np.zeros(shape = (self.n_train_seeds, n_values_1, n_values_2, n_seeds), dtype = np.float32)
        rarl_dataset = np.zeros(shape = (self.n_train_seeds, n_values_1, n_values_2, n_seeds), dtype = np.float32)

        shifted_env_config = deepcopy(self.env_config)

        shifted_env_config["additional"]["change"] = True

        for train_seed_iter in range(self.n_train_seeds):

            shifted_env_config["additional"]["params_to_change"] = {}
            
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(self.dir, str(train_seed))

            for value_1_iter in range(n_values_1):

                value_1 = values_1[value_1_iter]

                shifted_env_config["additional"]["params_to_change"][param_1] = value_1

                for value_2_iter in range(n_values_2):

                    value_2 = values_2[value_2_iter]

                    shifted_env_config["additional"]["params_to_change"][param_2] = value_2

                    for seed_iter in range(n_seeds):

                        seed = self.seeds[seed_iter]

                        shifted_env_config["seed"] = seed

                        shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                        shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                        shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                        vanilla_model_tester = ModelTester(
                            # algo = "PPO",
                            algo = self.env_config["algorithm"],
                            dir = os.path.join(curr_dir, "Vanilla"),
                            test_env = shifted_env_v,
                            norm = True,
                        )
                        vanilla_model_tester.load_last()

                        vanilla_dataset[train_seed_iter][value_1_iter][value_2_iter][seed_iter] = vanilla_model_tester.compute_reward(n_samples = 1)

                        arl_model_tester = ModelTester(
                            # algo = "PPO",
                            algo = self.env_config["algorithm"],
                            dir = os.path.join(curr_dir, "ARL"),
                            test_env = shifted_env_a,
                            norm = True,
                        )
                        arl_model_tester.load_last()

                        arl_dataset[train_seed_iter][value_1_iter][value_2_iter][seed_iter] = arl_model_tester.compute_reward(n_samples = 1)

                        rarl_model_tester = ModelTester(
                            # algo = "PPO",
                            algo = self.env_config["algorithm"],
                            dir = os.path.join(curr_dir, "RARL"),
                            test_env = shifted_env_r,
                            norm = True,
                        )
                        rarl_model_tester.load_last()

                        rarl_dataset[train_seed_iter][value_1_iter][value_2_iter][seed_iter] = rarl_model_tester.compute_reward(n_samples = 1)

        self.param_datasets[f"vanilla_{name}"] = vanilla_dataset
        self.param_datasets[f"arl_{name}"] = arl_dataset
        self.param_datasets[f"rarl_{name}"] = rarl_dataset

        self.param_datasets_dir = os.path.join(self.dir, f"datasets")
        os.makedirs(self.param_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.param_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "w") as file:
            json.dump(vanilla_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "w") as file:
            json.dump(arl_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "w") as file:
            json.dump(rarl_dataset.tolist(), file)

        



    def _plot_for_param_shift_two(
        self,
        name
    ):
        plot_dir = os.path.join(self.plot_dir, name)

        param_1, values_1 = self.param_shifted_envs[name][0]
        param_2, values_2 = self.param_shifted_envs[name][1]

        # create 2D values array
        values_1_grid, values_2_grid = np.meshgrid(values_1, values_2, indexing='ij')
        values = np.array([[ (values_1_grid[i,j], values_2_grid[i,j]) for j in range(values_2_grid.shape[1])] for i in range(values_1_grid.shape[0])])

        vanilla_dataset = self.param_datasets[f"vanilla_{name}"]
        arl_dataset = self.param_datasets[f"arl_{name}"]
        rarl_dataset = self.param_datasets[f"rarl_{name}"]

        vanilla_dataset = np.mean(vanilla_dataset, axis = -1)
        arl_dataset = np.mean(arl_dataset, axis = -1)
        rarl_dataset = np.mean(rarl_dataset, axis = -1)

        for train_seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(plot_dir, str(train_seed))

            # self._plot_2d(values, vanilla_dataset[train_seed_iter], "Vanilla", curr_dir, param_1, param_2)
            # self._plot_2d(values, arl_dataset[train_seed_iter], "Episode-Adv", curr_dir, param_1, param_2)
            # self._plot_2d(values, rarl_dataset[train_seed_iter], "Step-Adv", curr_dir, param_1, param_2)

            self._plot_2d(values, vanilla_dataset[train_seed_iter], arl_dataset[train_seed_iter], rarl_dataset[train_seed_iter], curr_dir, param_1, param_2, name)

        vanilla_dataset_mean = np.mean(vanilla_dataset, axis = 0)
        arl_dataset_mean = np.mean(arl_dataset, axis = 0)
        rarl_dataset_mean = np.mean(rarl_dataset, axis = 0)

        self.two_plot_name = name

        # self._plot_2d(values, vanilla_dataset_mean, "Vanilla", plot_dir, param_1, param_2)
        # self._plot_2d(values, arl_dataset_mean, "Episode-Adv", plot_dir, param_1, param_2)
        # self._plot_2d(values, rarl_dataset_mean, "Step-Adv", plot_dir, param_1, param_2)
        self._plot_2d(values, vanilla_dataset_mean, arl_dataset_mean, rarl_dataset_mean, plot_dir, param_1, param_2, name)


    def _plot_2d(
        self,
        values,
        vanilla_dataset,
        arl_dataset,
        rarl_dataset,
        # dataset_name,
        plot_dir,
        param_1,
        param_2,
        name
    ):
        # plt.style.use(['science', 'no-latex'])
        plt.style.use('science')

        plt.ioff()

        os.makedirs(plot_dir, exist_ok = True)

        X = values
        Y1 = vanilla_dataset
        Y2 = arl_dataset
        Y3 = rarl_dataset

        # # Flatten the X and Y arrays for plotting
        # x_coords = [coord[0] for row in X for coord in row]
        # y_coords = [coord[1] for row in X for coord in row]
        # y_values = [value for row in Y for value in row]

        # # Normalize Y values for color mapping
        # norm = plt.Normalize(min(y_values), max(y_values))
        # cmap = plt.cm.coolwarm  # Blue to red colormap

        # # Create the scatter plot and pass the colormap and normalization
        # scatter = plt.scatter(x_coords, y_coords, c=y_values, cmap=cmap, norm=norm, s=100)

        # # Add a color bar and associate it with the scatter plot
        # plt.colorbar(scatter, label='Value Spectrum')

        # # Increase margins to accommodate larger circles
        # plt.margins(0.2)

        # # Set axis labels and title
        # plt.xlabel(self.param_to_label[param_1])
        # plt.ylabel(self.param_to_label[param_2])
        # plt.title(f"{dataset_name}")

        # plt.savefig(f"{plot_dir}/{dataset_name}.png")
        
        # plt.close()

        plt.ioff()

        # Flatten the X coordinates for plotting
        x_coords = [coord[0] for row in X for coord in row]
        y_coords = [coord[1] for row in X for coord in row]

        # Flatten the Y arrays for plotting and find the global min and max for normalization
        y1_values = [value for row in Y1 for value in row]
        y2_values = [value for row in Y2 for value in row]
        y3_values = [value for row in Y3 for value in row]

        # # Combine all Y values to get global min and max
        # all_y_values = y1_values + y2_values + y3_values
        # global_min = min(all_y_values)
        # global_max = max(all_y_values)

        # # Flatten the Y arrays for plotting
        # y1_values = Y1.flatten()
        # y2_values = Y2.flatten()
        # y3_values = Y3.flatten()

        # Combine all Y values and get global min and max using np.min and np.max
        global_min = np.min([np.min(y1_values), np.min(y2_values), np.min(y3_values)])
        global_max = np.max([np.max(y1_values), np.max(y2_values), np.max(y3_values)])
        # global_max = np.max([y1_values.max(), y2_values.max(), y3_values.max()])

        # Normalize all Y values based on global min and max
        norm = plt.Normalize(global_min, global_max)
        cmap = plt.cm.coolwarm  # Blue to red colormap


        plt.rcParams.update({'font.size': 10})  # Set the global font size to 12

        plt.rcParams.update({
            'axes.labelsize': 11,    # Font size for x and y labels
            'xtick.labelsize': 10,   # Font size for x-axis tick labels
            'ytick.labelsize': 10,   # Font size for y-axis tick labels
            'axes.titlesize': 14     # Font size for the subplot titles
        })

        # Create the 1x3 subplot layout
        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        # plt.rcParams.update({'font.size': 10})  # Set the global font size to 12

        # plt.rcParams.update({
        #     'axes.labelsize': 10,    # Font size for x and y labels
        #     'xtick.labelsize': 10,   # Font size for x-axis tick labels
        #     'ytick.labelsize': 10,   # Font size for y-axis tick labels
        #     'axes.titlesize': 10     # Font size for the subplot titles
        # })

        # First plot (Y1)
        scatter1 = axs[0].scatter(x_coords, y_coords, c=y1_values, cmap=cmap, norm=norm, s=100)
        
        axs[0].set_xlabel(self.param_to_label[param_1])
        axs[0].set_ylabel(self.param_to_label[param_2])
        axs[0].margins(0.2)  # Increase margins
        axs[0].set_title("Vanilla")

        # Second plot (Y2)
        scatter2 = axs[1].scatter(x_coords, y_coords, c=y2_values, cmap=cmap, norm=norm, s=100)
        
        axs[1].set_xlabel(self.param_to_label[param_1])
        # axs[1].set_ylabel(self.param_to_label[param_2])
        axs[1].margins(0.2)  # Increase margins
        axs[1].set_title("Episode-Adv")

        # Third plot (Y3)
        scatter3 = axs[2].scatter(x_coords, y_coords, c=y3_values, cmap=cmap, norm=norm, s=100)
        
        axs[2].set_xlabel(self.param_to_label[param_1])
        # axs[2].set_ylabel(self.param_to_label[param_2])
        axs[2].margins(0.2)  # Increase margins
        axs[2].set_title("Step-Adv")

        # Add a color bar that applies to all subplots
        # Place the color bar to the right of the plots
        fig.subplots_adjust(right=0.85)  # Make room on the right for the colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='Reward Spectrum')

        # Adjust layout to ensure spacing
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the color bar on the right
        
        # plt.savefig(f"{plot_dir}/comparison.png", dpi = 300)
        # os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two/{self.two_plot_name}.png", dpi = 300)

        # os.makedirs(f"{self.dir}/robustness_comparison_plots/science-plots/fixed-two", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/science-plots/fixed-two/{self.two_plot_name}.png", dpi = 300)

        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two", exist_ok=True)
        plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two/{name}.png", dpi = 300)

        plt.close()


    def plot_existing_datasets_two_shifts(
        self,
        name,
        shifts
    ):
        self.param_shifted_envs[name] = []

        for shift in shifts:

            logspace = False
            if "logspace" in shift:
                logspace = shift["logspace"]

            low, high = shift["range"]
            n_values = shift["n_values"]
            param = shift["param"]

            if not logspace:
                values = np.linspace(low, high, n_values, dtype = np.float32)
            else:
                values = np.logspace(np.log10(low), np.log10(high), n_values, dtype = np.float32)

            if "values" in shift:
                values = shift["values"]
                if not isinstance(values, np.ndarray):
                    values = np.array(values)

            self.param_shifted_envs[name].append((param, values))



        self.two_plot_name = name

        self.param_datasets_dir = os.path.join(self.dir, f"datasets")
        os.makedirs(self.param_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.param_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "r") as file:
            # json.dump(vanilla_dataset.tolist(), file)
            vanilla_dataset = json.load(file)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "r") as file:
            # json.dump(arl_dataset.tolist(), file)
            arl_dataset = json.load(file)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "r") as file:
            # json.dump(rarl_dataset.tolist(), file)
            rarl_dataset = json.load(file)






        plot_dir = os.path.join(self.plot_dir, name)

        param_1, values_1 = self.param_shifted_envs[name][0]
        param_2, values_2 = self.param_shifted_envs[name][1]

        # create 2D values array
        values_1_grid, values_2_grid = np.meshgrid(values_1, values_2, indexing='ij')
        values = np.array([[ (values_1_grid[i,j], values_2_grid[i,j]) for j in range(values_2_grid.shape[1])] for i in range(values_1_grid.shape[0])])

        # vanilla_dataset = self.param_datasets[f"vanilla_{name}"]
        # arl_dataset = self.param_datasets[f"arl_{name}"]
        # rarl_dataset = self.param_datasets[f"rarl_{name}"]

        vanilla_dataset = np.mean(vanilla_dataset, axis = -1)
        arl_dataset = np.mean(arl_dataset, axis = -1)
        rarl_dataset = np.mean(rarl_dataset, axis = -1)

        for train_seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(plot_dir, str(train_seed))

            # self._plot_2d(values, vanilla_dataset[train_seed_iter], "Vanilla", curr_dir, param_1, param_2)
            # self._plot_2d(values, arl_dataset[train_seed_iter], "Episode-Adv", curr_dir, param_1, param_2)
            # self._plot_2d(values, rarl_dataset[train_seed_iter], "Step-Adv", curr_dir, param_1, param_2)

            self._plot_2d(values, vanilla_dataset[train_seed_iter], arl_dataset[train_seed_iter], rarl_dataset[train_seed_iter], curr_dir, param_1, param_2, name)

        vanilla_dataset_mean = np.mean(vanilla_dataset, axis = 0)
        arl_dataset_mean = np.mean(arl_dataset, axis = 0)
        rarl_dataset_mean = np.mean(rarl_dataset, axis = 0)

        # self._plot_2d(values, vanilla_dataset_mean, "Vanilla", plot_dir, param_1, param_2)
        # self._plot_2d(values, arl_dataset_mean, "Episode-Adv", plot_dir, param_1, param_2)
        # self._plot_2d(values, rarl_dataset_mean, "Step-Adv", plot_dir, param_1, param_2)
        self._plot_2d(values, vanilla_dataset_mean, arl_dataset_mean, rarl_dataset_mean, plot_dir, param_1, param_2, name)
        
    

    def add_adversarial_model(
        self,
        name,
        adv_model,
        adv_action_types
    ):
        self.adv_models[name] = (adv_model, adv_action_types)

        # test the three types of models (all three seeds each) against the stepwise adversary

        self._add_datasets_adv(name)

        self._plot_adv(name)

    
    def _add_datasets_adv(
        self,
        name
    ):
        adv_model, adv_action_types = self.adv_models[name]

        shifted_env_config = deepcopy(self.env_config)
        
        vanilla_dataset = np.zeros(shape = (self.n_train_seeds, self.n_seeds), dtype = np.float32)
        arl_dataset = np.zeros(shape = (self.n_train_seeds, self.n_seeds), dtype = np.float32)
        rarl_dataset = np.zeros(shape = (self.n_train_seeds, self.n_seeds), dtype = np.float32)

        for train_seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(self.dir, str(train_seed))

            for seed_iter in range(self.n_seeds):
                seed = self.seeds[seed_iter]

                shifted_env_config["seed"] = seed

                shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                vanilla_model_tester = ModelTester(
                    # algo = "PPO",
                    algo = self.env_config["algorithm"],
                    dir = os.path.join(curr_dir, "Vanilla"),
                    test_env = shifted_env_v,
                    norm = True,
                )
                vanilla_model_tester.load_last()

                vanilla_dataset[train_seed_iter][seed_iter] = vanilla_model_tester.compute_reward_with_adversary(
                    adv_model = adv_model,
                    n_samples = 1,
                    adv_action_types = adv_action_types
                )

                arl_model_tester = ModelTester(
                    # algo = "PPO",
                    algo = self.env_config["algorithm"],
                    dir = os.path.join(curr_dir, "ARL"),
                    test_env = shifted_env_a,
                    norm = True,
                )
                arl_model_tester.load_last()

                arl_dataset[train_seed_iter][seed_iter] = arl_model_tester.compute_reward_with_adversary(
                    adv_model = adv_model,
                    n_samples = 1,
                    adv_action_types = adv_action_types
                )

                rarl_model_tester = ModelTester(
                    # algo = "PPO",
                    algo = self.env_config["algorithm"],
                    dir = os.path.join(curr_dir, "RARL"),
                    test_env = shifted_env_r,
                    norm = True,
                )
                rarl_model_tester.load_last()

                rarl_dataset[train_seed_iter][seed_iter] = rarl_model_tester.compute_reward_with_adversary(
                    adv_model = adv_model,
                    n_samples = 1,
                    adv_action_types = adv_action_types
                )

        self.adv_datasets[f"vanilla_{name}"] = vanilla_dataset
        self.adv_datasets[f"arl_{name}"] = arl_dataset
        self.adv_datasets[f"rarl_{name}"] = rarl_dataset


        self.adv_datasets_dir = os.path.join(self.dir, f"adv_datasets")
        os.makedirs(self.adv_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.adv_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "w") as file:
            json.dump(vanilla_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "w") as file:
            json.dump(arl_dataset.tolist(), file)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "w") as file:
            json.dump(rarl_dataset.tolist(), file)

    def plot_existing_adv(
        self,
        name,
        adv_model,
        adv_action_types
    ):
        

        self.adv_datasets_dir = os.path.join(self.dir, f"adv_datasets")
        os.makedirs(self.adv_datasets_dir, exist_ok = True)

        this_dir = os.path.join(self.adv_datasets_dir, name)
        os.makedirs(this_dir, exist_ok = True)

        with open(file = os.path.join(this_dir, f"vanilla_rewards.json"), mode = "r") as file:
            vanilla_dataset = json.load(file)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "r") as file:
            arl_dataset = json.load(file)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "r") as file:
            rarl_dataset = json.load(file)

        self.adv_datasets[f"vanilla_{name}"] = vanilla_dataset
        self.adv_datasets[f"arl_{name}"] = arl_dataset
        self.adv_datasets[f"rarl_{name}"] = rarl_dataset

        self.adv_models[name] = (adv_model, adv_action_types)

        self._plot_adv(name)

    def format_param_string(self, param_string):
        # Split the input string based on '_'
        parts = param_string.split('_')
        
        # The first part is the parameter (convert it to lowercase)
        param = parts[0].lower()
        
        # The rest are the range values
        range_values = "-".join(parts[1:])
        
        # Format the output string using param_to_name dictionary
        return f"{self.param_to_name[param]}, {range_values}"
        
    def _plot_adv(
        self,
        name
    ):
        plot_dir = os.path.join(self.plot_dir, "adv")
        os.makedirs(plot_dir, exist_ok = True)
        
        vanilla_dataset = self.adv_datasets[f"vanilla_{name}"] 
        arl_dataset = self.adv_datasets[f"arl_{name}"]
        rarl_dataset = self.adv_datasets[f"rarl_{name}"]

        vanilla_dataset = np.mean(vanilla_dataset, axis = 1)
        arl_dataset = np.mean(arl_dataset, axis = 1)
        rarl_dataset = np.mean(rarl_dataset, axis = 1)

        # Calculate statistics
        means = [np.mean(vanilla_dataset), np.mean(arl_dataset), np.mean(rarl_dataset)]
        mins = [np.min(vanilla_dataset), np.min(arl_dataset), np.min(rarl_dataset)]
        maxs = [np.max(vanilla_dataset), np.max(arl_dataset), np.max(rarl_dataset)]

        # print(f"plotting for {self.dir[-11 :]} with adv {name}")

        # print(f"means - mins = \n{np.subtract(means, mins)}")

        # print(f"maxs - means = \n{np.subtract(maxs, means)}")

        means_sub_mins = np.subtract(means, mins)
        means_sub_mins = np.maximum(0, means_sub_mins)

        maxs_sub_means = np.subtract(maxs, means)
        maxs_sub_means = np.maximum(0, maxs_sub_means)

        plt.ioff()

        # Set up plot
        labels = ['Vanilla', 'Episode-Adv', 'Step-Adv']
        x = np.arange(len(labels))

        # Plot the data
        plt.figure(figsize=(4.5, 2.5))
        plt.errorbar(x, means, yerr=[means_sub_mins, maxs_sub_means], fmt='o', color='blue', capsize=5)
        # plt.errorbar(x, means, yerr=[np.subtract(means, mins), np.subtract(maxs, means)], fmt='o', color='blue', capsize=5)
        plt.xticks(x, labels, fontsize = 15)
        plt.ylabel('Reward', fontsize = 16)
        # plt.title('Reward by Strategy')

        plt.xlim(-0.5, len(x) - 0.5)
        plt.ylim(min(mins) - 5, max(maxs) + 5)

        plot_title = self.format_param_string(param_string=name)

        plt.title(plot_title, fontsize = 13)

        plt.tight_layout()

        plt.savefig(f"{plot_dir}/{name}.png", dpi = 300)
        
        plt.close()


            
    
        
    # MODEL LOADING HELPER FUNCTIONS
    
    def _make_vanilla_models(
        self
    ):
        self.vanilla_models = []
        for seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[seed_iter]
            this_vanilla_dir = os.path.join(self.dir, f"{train_seed}/Vanilla")
            with open(file = os.path.join(this_vanilla_dir, "saved/model_config.json"), mode = "r") as file:
                this_vanilla_model_config = json.load(file)

            if "model_dir" in this_vanilla_model_config:
                del this_vanilla_model_config["model_dir"]
            if "env_dir" in this_vanilla_model_config:
                del this_vanilla_model_config["env_dir"]
            
            self.vanilla_models.append(VanillaModelManager(**this_vanilla_model_config))

            

            self.vanilla_models[seed_iter].load()

    
    def _make_arl_models(
        self
    ):
        self.arl_models = []
        for seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[seed_iter]
            this_arl_dir = os.path.join(self.dir, f"{train_seed}/ARL")
            with open(file = os.path.join(this_arl_dir, "saved/model_config.json"), mode = "r") as file:
                this_arl_model_config = json.load(file)
            
            if "model_dir" in this_arl_model_config:
                del this_arl_model_config["model_dir"]
            if "env_dir" in this_arl_model_config:
                del this_arl_model_config["env_dir"]
            if "env_config" in this_arl_model_config:
                del this_arl_model_config["env_config"]

            self.arl_models.append(ARLModelManager(**this_arl_model_config))
            self.arl_models[seed_iter].load()

    def _make_rarl_models(
        self
    ):
        self.rarl_models = []
        for seed_iter in range(self.n_train_seeds):
            train_seed = self.train_seeds[seed_iter]
            this_rarl_dir = os.path.join(self.dir, f"{train_seed}/RARL")
            with open(file = os.path.join(this_rarl_dir, "saved/model_config.json"), mode = "r") as file:
                this_rarl_model_config = json.load(file)

            if "model_dir" in this_rarl_model_config:
                del this_rarl_model_config["model_dir"]
            if "env_dir" in this_rarl_model_config:
                del this_rarl_model_config["env_dir"]
            if "env_config" in this_rarl_model_config:
                del this_rarl_model_config["env_config"]

            self.rarl_models.append(RARLModelManager(**this_rarl_model_config))
            self.rarl_models[seed_iter].load()
        
        


class RobustnessEvaluation:

    """
    args:
        dir: the dir
        n_samples: number of sampled episdoes per plot point
        plot_freq: number of models left before next plot point

    TODO: needs to have following functionality:
        - take as input 
            - train env/its config
            - directory of model(s)
        - have an "add_robust_env" function that takes as input (type of shift, shift parameters), creates the shifted env, tests both the rarl and vanilla models on the shifted env, and makes the following graphs
            - rarl vs vanilla for each shifted env
            - rarl on shifted env vs rarl on train env for each shifted env
            - vanilla on shifted env vs vanilla on train env for each shifted env
            - rarl vs vanilla for each shifted env
            - rarl on shifted vs rarl on train vs vanilla on shifted vs vanilla on train for each shifted env
        - have an add plot feature that can take in a plot type and retrospectively draw the plot for each shifted env in the object
    """

    def __init__(
        self,
        dir = None,
        n_samples: int = 3,
        plot_freq: int = 5
        # train_env_config = {},
        # train_env_class = VanillaGovABC
    ):
        if dir == None:
            raise ValueError("dir cannot be None")
        self.dir = dir

        self.n_samples = n_samples
        self.plot_freq = plot_freq
        
        # if train_env_class == None:
        #     raise ValueError("train_env_class cannot be None")
        # self.train_env_class = train_env_class
        
        # if train_env_config == None:
        #     raise ValueError("train_env_config cannot be none")
        # self.train_env_config = train_env_config
        
        # self.train_env = self.train_env_class(**self.train_env_config)

        # creating RARL model
        rarl_dir = os.path.join(self.dir, "RARL")
        # print(f"rarl_dir is: \n{rarl_dir}\n")
        with open(os.path.join(rarl_dir, "saved/model_config.json"), "r") as file:
            rarl_model_config = json.load(file)
        
        # TODO: remove for later experiments
        if "env_config" in rarl_model_config:
            del rarl_model_config["env_config"]

        self.rarl_model = RARLModelManager(**rarl_model_config)
        self.rarl_model.load(os.path.join(rarl_dir, "saved"))

        # creating Vanilla model
        vanilla_dir = os.path.join(self.dir, "Vanilla")
        with open(os.path.join(vanilla_dir, "saved/model_config.json"), "r") as file:
            vanilla_model_config = json.load(file)
        self.vanilla_model = VanillaModelManager(**vanilla_model_config)
        self.vanilla_model.load(os.path.join(vanilla_dir, "saved"))

        self.train_env = deepcopy(self.vanilla_model.train_env)
        self.train_env_config = vanilla_model_config["env_config"]
        self.train_env_class = self.vanilla_model.env_class

        self.shifted_envs = {"same" : deepcopy(self.train_env)}

        self.plot_types = []
        
        self.datasets = {}

        self.shifted_envs_range = {}

        self.datasets_range = {}

        # dir variables
        
        self.plot_dir = os.path.join(self.dir, "plots")
        os.makedirs(self.plot_dir, exist_ok = True)

        self.vanilla_dir = os.path.join(self.dir, "Vanilla")
        self.rarl_dir = os.path.join(self.dir, "RARL")

        self.vanilla_model_dir = os.path.join(self.vanilla_dir, "models")
        self.rarl_model_dir = os.path.join(self.rarl_dir, "models")


        # model variables for plotting
        self.n_eps_vanilla = int(self.vanilla_model.model.num_timesteps / self.vanilla_model.episode_length)
        self.n_eps_rarl = int(self.rarl_model.gov_model.num_timesteps / self.rarl_model.episode_length)
        self.update_freq_vanilla = self.vanilla_model.update_freq
        self.update_freq_rarl = self.rarl_model.update_freq
        self.n_vanilla_models = int(self.vanilla_model.model.num_timesteps / self.vanilla_model.n_steps)
        self.n_rarl_models = int(self.rarl_model.gov_model.num_timesteps / self.rarl_model.n_steps)
        self.vanilla_to_norm = os.path.exists(os.path.join(self.vanilla_dir, "envs/1.pkl"))
        self.rarl_to_norm = os.path.exists(os.path.join(self.rarl_dir, "envs/1.pkl"))
        self.vanilla_episode_checkpoint_multiplier = int(self.vanilla_model.n_steps / self.vanilla_model.episode_length)
        self.rarl_episode_checkpoint_multiplier = int(self.rarl_model.n_steps / self.rarl_model.episode_length)

        # plot variables
        self.vanilla_episodes_axis = np.array(
            [(iter + 1) * int(self.vanilla_model.n_steps / self.vanilla_model.episode_length) * self.plot_freq for iter in range(int(self.n_vanilla_models / self.plot_freq))]
        )

        self.rarl_episodes_axis = np.array(
            [(iter + 1) * int(self.rarl_model.n_steps / self.rarl_model.episode_length) * self.plot_freq for iter in range(int(self.n_rarl_models / self.plot_freq))]
        )

        # self.vanilla_episodes_axis = np.array(
        #     [(iter + 1) * int(self.vanilla_model.n_steps / self.vanilla_model.episode_length) for iter in range(self.n_vanilla_models)]
        # )
        
        # self.rarl_episodes_axis = np.array(
        #     [(iter + 1) * int(self.rarl_model.n_steps / self.rarl_model.episode_length) for iter in range(self.n_rarl_models)]
        # )

        

        self._add_basic_plot_types()

        self._add_datasets("same", self.train_env)
        self._plot_for_env("same")

        self._add_basic_shifted_envs()

    def add_shifted_env(
        self,
        shift_types: list = ["behavioural"],
        shift: dict = {},
        env_name = None
    ):
        """

        args:
            - shift_type: type of shift. 
                behavioural: change some behavioural parameter in the model (before burn in) 
                scale: change the scale of the model

        TODO:
            - have a list of all the shifted envs and a good way to access them
            - have a list of the plot dirs for each shifted env
        """

        if env_name == None:
            env_name = "shifted_env_" + str(len(self.shifted_envs) + 1)

        shifted_env_config = deepcopy(self.train_env_config)

        if "behavioural" in shift_types:

            shifted_env_config["additional"]["change"] = True

            # shift has a key called "shifted_params" which is a list of the names of the shifted parameters
            # shift[<name of shifted param>] stores the shifted value of the shifted param

            shifted_env_config["additional"]["params_to_change"] = {}
            for param in shift["shifted_params"]:
                shifted_env_config["additional"]["params_to_change"][param] = shift[param]

            
        if "scale" in shift_types:
            
            shifted_env_config["W"] = int(shift["scale"] * shifted_env_config["W"])
            shifted_env_config["F"] = int(shift["scale"] * shifted_env_config["F"])
            shifted_env_config["N"] = int(shift["scale"] * shifted_env_config["N"])

        shifted_env = self.train_env_class(**shifted_env_config)

        self.shifted_envs[env_name] = shifted_env

        self._add_datasets(env_name, shifted_env)

        self._plot_for_env(env_name)

        self.save()

    def add_shifted_env_range(
        self,
        shift_types = ["behavioural"],
        shift: dict = {},
        name: str = None
    ):
        
        if name == None:
            name = "shifted_env_range_" + str(len(self.shifted_envs_range) + 1)
        
        shifted_env_config = deepcopy(self.train_env_config)

        n_models = 1

        if "behavioural" in shift_types:

            # n_values
            n_values = shift["n_values"]
            n_models = shift["n_models"]
            # fip or cwr or mp
            param = shift["param"]

            low = shift["low"]
            high = shift["high"]

            values = [low + (iter * (high - low) / n_values) for iter in range(n_values + 1)]

            # self.shifted_envs_range[param] = {}

            self.shifted_envs_range[name] = (param, values)

        self._add_datasets_range(name, n_models)

        self._plot_for_range(name)

        self.save()

        # "shift" contains multiple params
        #   (as of now we assume shift_types = ["behavioural"])
        #   

    def _add_datasets_range(
        self,
        name,
        n_models
    ):
        shifted_env_config = deepcopy(self.train_env_config)

        shifted_env_config["additional"]["change"] = True

        param = self.shifted_envs_range[name][0]
        values = self.shifted_envs_range[name][1]

        vanilla_dataset = np.zeros(shape = (len(values), n_models), dtype = np.float32)

        rarl_dataset = np.zeros(shape = (len(values), n_models), dtype = np.float32)

        for value_iter in range(len(values)):

            value = values[value_iter]
            
            shifted_env_config["additional"]["params_to_change"] = {param: value}

            shifted_env = self.train_env_class(**shifted_env_config)

            vanilla_tester = ModelTester(
                algo = self.vanilla_model.algorithm, 
                dir = self.vanilla_dir, 
                norm = self.vanilla_to_norm, 
                test_env = shifted_env
            )

            rarl_tester = ModelTester(
                algo = self.rarl_model.gov_algorithm,
                dir = self.rarl_dir,
                norm = self.rarl_to_norm,
                test_env = shifted_env
            )

            # vanilla_dataset = np.zeros(shape = (n_models,), dtype = np.float32)

            for iter in range(n_models):

                vanilla_tester.load(
                    checkpoint_no = int(iter + self.n_vanilla_models - n_models + 1)
                )

                vanilla_dataset[value_iter][iter] = vanilla_tester.compute_reward(n_samples = self.n_samples)

                rarl_tester.load(
                    checkpoint_no = int(iter + self.n_rarl_models - n_models + 1)
                )

                rarl_dataset[value_iter][iter] = rarl_tester.compute_reward(n_samples = self.n_samples)
            
        self.datasets_range[f"vanilla_{name}"] = vanilla_dataset
        
        self.datasets_range[f"rarl_{name}"] = rarl_dataset


        with open(
            file = os.path.join(
                self.vanilla_dir, 
                f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{len(values)}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(vanilla_dataset.tolist(), file)

    
        with open(
            file = os.path.join(
                self.rarl_dir, 
                f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{len(values)}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(rarl_dataset.tolist(), file)


    def _plot_for_range(
        self,
        name
    ):
        dir = os.path.join(self.plot_dir, "range plots")
        
        values = self.shifted_envs_range[name][1]
        param = values = self.shifted_envs_range[name][0]
        
        vanilla_dataset = self.datasets_range[f"vanilla_{name}"]
        rarl_dataset = self.datasets_range[f"rarl_{name}"]

        vanilla_means = np.mean(vanilla_dataset, axis = 1)
        vanilla_std_devs = np.std(vanilla_dataset, axis = 1)

        rarl_means = np.mean(rarl_dataset, axis = 1)
        rarl_std_devs = np.std(rarl_dataset, axis = 1)

        plt.ioff()

        plt.errorbar(values, vanilla_means, yerr = vanilla_std_devs, fmt = "0", color = "green", ecolor = "green", capsize = 5)
        plt.plot(values, vanilla_means, color = "green", label = "Vanilla")

        plt.errorbar(values, rarl_means, yerr = rarl_std_devs, fmt = "0", color = "blue", ecolor = "blue", capsize = 5)
        plt.plot(values, rarl_means, color = "blue", label = "RARL")

        plt.xlabel(param)
        plt.ylabel("Rewards")

        plt.title(f"Rewards on {param} from {values[0]} to {values[-1]}")

        plt.legend()

        plt.savefig(os.path.join(dir, f"{name}_{values[0]}_to_{values[-1]}"))
        
        plt.close()

        



    def add_plot_type():
        """
        TODO:
            - at the end, add a call to a function that does something like "retroactively plot for all"
        """
        pass

    def save(self):

        save_dir = os.path.join(self.dir, "saved")
        os.makedirs(save_dir, exist_ok = True)

        with open(os.path.join(save_dir, "evaluator.pkl"), "wb") as file:
            pickle.dump(self, file)
        
        file_path = os.path.join(save_dir, "evaluator.pkl")

        print(f"RobustnessEvaluation object saved to {file_path}")
        

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
        
        print(f"RobustnessEvaluation object loaded from {file_path}")
        
        return obj



    def _plot_for_env(
        self,
        env_name
    ):
        """
        makes the following graphs
            - rarl vs vanilla for each shifted env
            - rarl on shifted env vs rarl on train env for each shifted env
            - vanilla on shifted env vs vanilla on train env for each shifted env
            - rarl vs vanilla for each shifted env
            - rarl on shifted vs rarl on train vs vanilla on shifted vs vanilla on train for each shifted env
        """
        this_plot_dir = os.path.join(self.plot_dir, env_name)
        os.makedirs(this_plot_dir, exist_ok = True)

        env = self.shifted_envs[env_name]
        
        for plot_function in self.plot_types:
            
            plot_function(
                env_name,
                this_plot_dir
            )
    
    def _add_datasets(
        self,
        env_name,
        env
    ):
        """
        add vanilla and rarl datasets on env
        """
        print()
        print(f"adding datasets for {env_name}")

        test_env = deepcopy(env)

        vanilla_tester = ModelTester(
            algo = self.vanilla_model.algorithm,
            dir = self.vanilla_dir,
            norm = self.vanilla_to_norm,
            test_env = test_env
        )

        vanilla_dataset = np.zeros(shape = (self.vanilla_episodes_axis.shape[0]), dtype = np.float32)

        print("comptuations for vanilla dataset starts")
        for iter in range(int(self.n_vanilla_models / self.plot_freq)):
            vanilla_tester.load(
                checkpoint_no = int(self.vanilla_episodes_axis[iter] / self.vanilla_episode_checkpoint_multiplier)
            )
            vanilla_dataset[iter] = vanilla_tester.compute_reward(n_samples = self.n_samples)
            # print(f"iter {iter+1} i.e. model {int(self.vanilla_episodes_axis[iter] / self.vanilla_episode_checkpoint_multiplier)} computed")
        print("comptuations for vanilla dataset ends")

        self.datasets[f"vanilla_{env_name}"] = vanilla_dataset

        rarl_tester = ModelTester(
            algo = self.rarl_model.gov_algorithm,
            dir = self.rarl_dir,
            norm = self.rarl_to_norm,
            test_env = test_env
        )

        rarl_dataset = np.zeros(shape = (self.rarl_episodes_axis.shape[0]), dtype = np.float32)

        print("comptuations for rarl dataset starts")
        for iter in range(int(self.n_rarl_models / self.plot_freq)):
            rarl_tester.load(
                checkpoint_no = int(self.rarl_episodes_axis[iter] / self.rarl_episode_checkpoint_multiplier)
            )
            rarl_dataset[iter] = rarl_tester.compute_reward(n_samples = self.n_samples)
            # print(f"iter {iter+1} i.e. model {int(self.rarl_episodes_axis[iter] / self.rarl_episode_checkpoint_multiplier)} computed")
        print("comptuations for rarl dataset ends")

        self.datasets[f"rarl_{env_name}"] = rarl_dataset

        # save datasets for later
        print("saving datasets")
        with open(
            file = os.path.join(
                self.vanilla_dir, 
                f"data/rewards_on_{env_name}_plot_freq_{self.plot_freq}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(vanilla_dataset.tolist(), file)

        with open(
            file = os.path.join(
                self.rarl_dir, 
                f"data/rewards_on_{env_name}_plot_freq_{self.plot_freq}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(rarl_dataset.tolist(), file)
        
        print("datasets saved")
        print(f"computations for {env_name} ended")
        
    def _add_basic_shifted_envs(
        self
    ):
        shift1_types = ["scale"]
        shift1 = {"scale" : 0.2}
        shift1_name = "scale_0.2x"
        self.add_shifted_env(shift1_types, shift1, shift1_name)

        # shift2_types = ["scale"]
        # shift2 = {"scale" : 4.0}
        # shift2_name = "scale_4x"
        # self.add_shifted_env(shift2_types, shift2, shift2_name)

        # shift3_types = ["behavioural"]
        # shift3 = {
        #     "shifted_params" : ["consumption_wealth_ratio"],
        #     "consumption_wealth_ratio" : 0.0
        # }
        # shift3_name = "cwr_0.0"
        # self.add_shifted_env(shift3_types, shift3, shift3_name)

        # shift4_types = ["behavioural"]
        # shift4 = {
        #     "shifted_params" : ["firm_invest_prob"],
        #     "firm_invest_prob" : 0.0
        # }
        # shift4_name = "fip_0.0"
        # self.add_shifted_env(shift4_types, shift4, shift4_name)
    
    def _add_basic_plot_types(
        self
    ):
        """
        makes the following graphs
            - rarl vs vanilla for each shifted env
            - rarl on shifted env vs rarl on train env for each shifted env
            - vanilla on shifted env vs vanilla on train env for each shifted env
            - rarl on shifted vs rarl on train vs vanilla on shifted vs vanilla on train for each shifted env
        note that each graph is a reward vs episodes graph
        
        each model checkpoint should be loaded on its env checkpoint
        """

        self.plot_types.append(self._plot_rarl_and_vanilla_shifted_vs_train)
        self.plot_types.append(self._plot_rarl_shifted_vs_train)
        self.plot_types.append(self._plot_vanilla_shifted_vs_train)
        self.plot_types.append(self._plot_rarl_vs_vanilla)

    def _plot_rarl_and_vanilla_shifted_vs_train(
        self,
        env_name,
        dir
    ):
        
        vanilla_shifted_dataset = self.datasets[f"vanilla_{env_name}"]
        rarl_shifted_dataset = self.datasets[f"rarl_{env_name}"]

        vanilla_same_dataset = self.datasets["vanilla_same"]
        rarl_same_dataset = self.datasets["rarl_same"]

        plt.ioff()

        plt.plot(self.vanilla_episodes_axis, vanilla_shifted_dataset, color = "limegreen", label = f"vanilla_{env_name}")
        plt.plot(self.vanilla_episodes_axis, vanilla_same_dataset, color = "green", label = "vanilla_same")
        plt.plot(self.rarl_episodes_axis, rarl_shifted_dataset, color = "cyan", label = f"rarl_{env_name}")
        plt.plot(self.rarl_episodes_axis, rarl_same_dataset, color = "blue", label = "rarl_same")

        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(f"RARL vs Vanilla on {env_name} and same envs")

        plt.legend()

        plt.savefig(os.path.join(dir, f"rarl_and_vanilla_{env_name}_vs_same.png"))

        plt.close()

    def _plot_vanilla_shifted_vs_train(
        self,
        env_name,
        dir
    ):
        
        vanilla_shifted_dataset = self.datasets[f"vanilla_{env_name}"]
        vanilla_same_dataset = self.datasets["vanilla_same"]

        plt.ioff()

        plt.plot(self.vanilla_episodes_axis, vanilla_shifted_dataset, color = "limegreen", label = f"vanilla_{env_name}")
        plt.plot(self.vanilla_episodes_axis, vanilla_same_dataset, color = "green", label = "vanilla_same")

        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(f"Vanilla on {env_name} and same envs")

        plt.legend()

        plt.savefig(os.path.join(dir, f"vanilla_{env_name}_vs_same.png"))

        plt.close()

    def _plot_rarl_shifted_vs_train(
        self,
        env_name,
        dir
    ):
        rarl_shifted_dataset = self.datasets[f"rarl_{env_name}"]
        rarl_same_dataset = self.datasets["rarl_same"]

        plt.ioff()

        plt.plot(self.rarl_episodes_axis, rarl_shifted_dataset, color = "cyan", label = f"rarl_{env_name}")
        plt.plot(self.rarl_episodes_axis, rarl_same_dataset, color = "blue", label = "rarl_same")

        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(f"RARL on {env_name} and same envs")

        plt.legend()

        plt.savefig(os.path.join(dir, f"rarl_{env_name}_vs_same.png"))

        plt.close()


    def _plot_rarl_vs_vanilla(
        self,
        env_name,
        dir
    ):
        
        vanilla_shifted_dataset = self.datasets[f"vanilla_{env_name}"]
        rarl_shifted_dataset = self.datasets[f"rarl_{env_name}"]

        plt.ioff()

        plt.plot(self.vanilla_episodes_axis, vanilla_shifted_dataset, color = "limegreen", label = f"vanilla_{env_name}")
        plt.plot(self.rarl_episodes_axis, rarl_shifted_dataset, color = "cyan", label = f"rarl_{env_name}")

        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.title(f"RARL vs Vanilla on {env_name} envs")

        plt.legend()

        plt.savefig(os.path.join(dir, f"rarl_and_vanilla_{env_name}.png"))

        plt.close()


    # def _plot_rarl_and_vanilla_shifted_vs_train(
    #     self,
    #     env,
    #     name,
    #     dir
    # ):
    #     plt.ioff()
    #     vanilla_shifted_tester = ModelTester(
    #         algo = self.vanilla_model.algorithm,
    #         dir = self.vanilla_dir,
    #         norm = self.vanilla_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     vanilla_shifted_rewards_axis = np.zeros(shape = (self.n_vanilla_models), dtype = np.float32)

    #     for iter in range(self.n_rarl_models):
    #         vanilla_shifted_tester.load(checkpoint_no = iter + 1)
    #         vanilla_shifted_rewards_axis[iter] = vanilla_shifted_tester.compute_reward(n_samples = 2)

    #     vanilla_same_tester = ModelTester(
    #         algo = self.vanilla_model.algorithm,
    #         dir = self.vanilla_dir,
    #         norm = self.vanilla_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(self.train_env)
    #     )

    #     same_rewards_axis = np.zeros(shape = (self.n_vanilla_models), dtype = np.float32)
        
    #     for iter in range(self.n_vanilla_models):
    #         vanilla_same_tester.load(checkpoint_no = iter + 1)
    #         same_rewards_axis[iter] = vanilla_same_tester.compute_reward(n_samples = 2)
    #     # this_plot_dir = os.path.join(self.plot_dir, name), exist_ok = True
    #     # os.makedirs(this_plot_dir)
        
    #     plt.plot(
    #         self.vanilla_episodes_axis, 
    #         vanilla_shifted_rewards_axis,
    #         label = f"vanilla_{name}",
    #         color = "limegreen"
    #     )
        
    #     plt.plot(
    #         self.vanilla_episodes_axis,
    #         same_rewards_axis,
    #         label = "same",
    #         color = "green"
    #     )

    #     rarl_shifted_tester = ModelTester(
    #         algo = self.rarl_model.gov_algorithm,
    #         dir = self.rarl_dir,
    #         norm = self.rarl_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     rarl_shifted_rewards_axis = np.zeros(shape = (self.n_rarl_models), dtype = np.float32)

    #     for iter in range(self.n_rarl_models):

    #         rarl_shifted_tester.load(checkpoint_no = iter + 1)
    #         rarl_shifted_rewards_axis[iter] = rarl_shifted_tester.compute_reward(n_samples = 2)

    #     rarl_same_tester = ModelTester(
    #         algo = self.rarl_model.gov_algorithm,
    #         dir = self.rarl_dir,
    #         norm = self.rarl_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(self.train_env)
    #     )

    #     same_rewards_axis = np.zeros(shape = (self.n_rarl_models), dtype = np.float32)
        
    #     # for rarl
    #     for iter in range(self.n_rarl_models):

    #         rarl_same_tester.load(checkpoint_no = iter + 1)
    #         same_rewards_axis[iter] = rarl_same_tester.compute_reward(n_samples = 2)

    #     # this_plot_dir = os.path.join(self.plot_dir, name), exist_ok = True
    #     # os.makedirs(this_plot_dir)
        
    #     plt.plot(
    #         self.rarl_episodes_axis, 
    #         rarl_shifted_rewards_axis,
    #         label = f"rarl_{name}",
    #         color = "cyan"
    #     )
        
    #     plt.plot(
    #         self.rarl_episodes_axis, 
    #         same_rewards_axis,
    #         label = "rarl_same",
    #         color = "blue"
    #     )
        
    #     plt.ylabel("Rewards")
    #     plt.xlabel("Episodes")
    #     plt.title(f"RARL and Vanilla models on {name} env vs same env")
    #     plt.legend()
    #     plt.savefig(os.path.join(dir, f"rarl_and_vanilla_{name}_vs_same.png"))
    #     plt.close()
    

    
    # def _plot_vanilla_shifted_vs_train(
    #     self,
    #     env,
    #     name,
    #     dir
    # ):
    #     plt.ioff()
    #     vanilla_shifted_tester = ModelTester(
    #         algo = self.vanilla_model.algorithm,
    #         dir = self.vanilla_dir,
    #         norm = self.vanilla_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     shifted_rewards_axis = np.zeros(shape = (self.n_vanilla_models), dtype = np.float32)

    #     for iter in range(self.n_rarl_models):

    #         vanilla_shifted_tester.load(checkpoint_no = iter + 1)
    #         shifted_rewards_axis[iter] = vanilla_shifted_tester.compute_reward(n_samples = 2)

    #     vanilla_same_tester = ModelTester(
    #         algo = self.vanilla_model.algorithm,
    #         dir = self.vanilla_dir,
    #         norm = self.vanilla_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(self.train_env)
    #     )

    #     same_rewards_axis = np.zeros(shape = (self.n_vanilla_models), dtype = np.float32)
        
    #     # for rarl
    #     for iter in range(self.n_vanilla_models):

    #         vanilla_same_tester.load(checkpoint_no = iter + 1)
    #         same_rewards_axis[iter] = vanilla_same_tester.compute_reward(n_samples = 2)

    #     # this_plot_dir = os.path.join(self.plot_dir, name), exist_ok = True
    #     # os.makedirs(this_plot_dir)
        
    #     plt.plot(
    #         self.vanilla_episodes_axis, 
    #         shifted_rewards_axis,
    #         label = name,
    #         color = "limegreen"
    #     )
        
    #     plt.plot(
    #         self.vanilla_episodes_axis, 
    #         same_rewards_axis,
    #         label = "same",
    #         color = "green"
    #     )

    #     plt.ylabel("Rewards")
    #     plt.xlabel("Episodes")
    #     plt.title(f"Vanilla model on {name} env vs same env")
    #     plt.legend()
    #     plt.savefig(os.path.join(dir, f"vanilla_{name}_vs_same.png"))
    #     plt.close()

    # def _plot_rarl_shifted_vs_train(
    #     self,
    #     env,
    #     name,
    #     dir
    # ):
    #     plt.ioff()
    #     rarl_shifted_tester = ModelTester(
    #         algo = self.rarl_model.gov_algorithm,
    #         dir = self.rarl_dir,
    #         norm = self.rarl_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     shifted_rewards_axis = np.zeros(shape = (self.n_rarl_models), dtype = np.float32)

    #     for iter in range(self.n_rarl_models):

    #         rarl_shifted_tester.load(checkpoint_no = iter + 1)
    #         shifted_rewards_axis[iter] = rarl_shifted_tester.compute_reward(n_samples = 2)

    #     rarl_same_tester = ModelTester(
    #         algo = self.rarl_model.gov_algorithm,
    #         dir = self.rarl_dir,
    #         norm = self.rarl_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(self.train_env)
    #     )

    #     same_rewards_axis = np.zeros(shape = (self.n_rarl_models), dtype = np.float32)
        
    #     # for rarl
    #     for iter in range(self.n_rarl_models):

    #         rarl_same_tester.load(checkpoint_no = iter + 1)
    #         same_rewards_axis[iter] = rarl_same_tester.compute_reward(n_samples = 2)

    #     # this_plot_dir = os.path.join(self.plot_dir, name), exist_ok = True
    #     # os.makedirs(this_plot_dir)
        
    #     plt.plot(
    #         self.rarl_episodes_axis, 
    #         shifted_rewards_axis,
    #         label = name,
    #         color = "cyan"
    #     )
        
    #     plt.plot(
    #         self.rarl_episodes_axis, 
    #         same_rewards_axis,
    #         label = "same",
    #         color = "blue"
    #     )

    #     plt.ylabel("Rewards")
    #     plt.xlabel("Episodes")
    #     plt.title(f"RARL model on {name} env vs same env")
    #     plt.legend()
    #     plt.savefig(os.path.join(dir, f"rarl_{name}_vs_same.png"))
    #     plt.close()


    # def _plot_rarl_vs_vanilla(
    #     self,
    #     env,
    #     name,
    #     dir
    # ):
    #     plt.ioff()
    #     # first make the x and y arrays
    #     #   x array is the episodes axis, which is already given
    #     #   y arrays are the reward arrays on the env for rarl and vanilla models

    #     vanilla_model_tester = ModelTester(
    #         algo = self.vanilla_model.algorithm,
    #         dir = self.vanilla_dir,
    #         norm = self.vanilla_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     vanilla_rewards_axis = np.zeros(shape = (self.n_vanilla_models), dtype = np.float32)

    #     rarl_model_tester = ModelTester(
    #         algo = self.rarl_model.gov_algorithm,
    #         dir = self.rarl_dir,
    #         norm = self.rarl_to_norm,  # TODO: add norm_reward to this as well (in ModelTester)
    #         test_env = deepcopy(env)
    #     )

    #     rarl_rewards_axis = np.zeros(shape = (self.n_rarl_models), dtype = np.float32)

    #     # creating y array for vanilla
    #     for iter in range(self.n_vanilla_models):

    #         vanilla_model_tester.load(checkpoint_no = iter + 1)
    #         vanilla_rewards_axis[iter] = vanilla_model_tester.compute_reward(n_samples = 2)
        
    #     # for rarl
    #     for iter in range(self.n_rarl_models):

    #         rarl_model_tester.load(checkpoint_no = iter + 1)
    #         rarl_rewards_axis[iter] = rarl_model_tester.compute_reward(n_samples = 2)

    #     # this_plot_dir = os.path.join(self.plot_dir, name), exist_ok = True
    #     # os.makedirs(this_plot_dir)
        
    #     plt.plot(
    #         self.vanilla_episodes_axis, 
    #         vanilla_rewards_axis,
    #         label = "Vanilla",
    #         color = "green"
    #     )
        
    #     plt.plot(
    #         self.rarl_episodes_axis, 
    #         rarl_rewards_axis,
    #         label = "RARL",
    #         color = "blue"
    #     )

    #     plt.ylabel("Rewards")
    #     plt.xlabel("Episodes")
    #     plt.title(f"RARL vs Vanilla on {name} env")
    #     plt.legend()
    #     plt.savefig(os.path.join(dir, f"{name}_rarl_vs_vanilla.png"))
    #     plt.close()


        
    # def _two_plots_helper(
    #     self,
    #     x1,
    #     x2,
    #     y1,
    #     y2
    # ):
    #     """
    #     x1 and x2 are assumed to be episode axes of rarl and vanilla; albeit, can be of different lengths
    #     hence, we need to plot such that both of them can be in one plot
    #     """
    #     pass

    # def _add_basic_datasets(
    #     self
    # ):
    #     """
    #     basic datasets are:
    #         rarl on train env
    #         vanilla on train env
    #         rarl on each shifted env
    #         vanilla on each shifted env
    #     """


class RobustnessEvaluationRange:
    def __init__(
        self,
        dir = None,
        n_samples: int = 3,
    ):
        if dir == None:
            raise ValueError("dir cannot be None")
        self.dir = dir

        self.n_samples = n_samples
    
        # creating RARL model
        rarl_dir = os.path.join(self.dir, "RARL")
        with open(os.path.join(rarl_dir, "saved/model_config.json"), "r") as file:
            rarl_model_config = json.load(file)
        
        # TODO: remove for later experiments
        if "env_config" in rarl_model_config:
            del rarl_model_config["env_config"]

        self.rarl_model = RARLModelManager(**rarl_model_config)
        self.rarl_model.load(os.path.join(rarl_dir, "saved"))

        # creating Vanilla model
        vanilla_dir = os.path.join(self.dir, "Vanilla")
        with open(os.path.join(vanilla_dir, "saved/model_config.json"), "r") as file:
            vanilla_model_config = json.load(file)
        self.vanilla_model = VanillaModelManager(**vanilla_model_config)
        self.vanilla_model.load(os.path.join(vanilla_dir, "saved"))

        self.train_env = deepcopy(self.vanilla_model.train_env)
        self.train_env_config = vanilla_model_config["env_config"]
        self.train_env_class = self.vanilla_model.env_class

        self.shifted_envs_range = {}

        self.datasets_range = {}

        # dir variables
        
        self.plot_dir = os.path.join(self.dir, "range_plots")
        os.makedirs(self.plot_dir, exist_ok = True)

        self.vanilla_dir = os.path.join(self.dir, "Vanilla")
        self.rarl_dir = os.path.join(self.dir, "RARL")

        self.vanilla_model_dir = os.path.join(self.vanilla_dir, "models")
        self.rarl_model_dir = os.path.join(self.rarl_dir, "models")


        # model variables for plotting
        self.n_eps_vanilla = int(self.vanilla_model.model.num_timesteps / self.vanilla_model.episode_length)
        self.n_eps_rarl = int(self.rarl_model.gov_model.num_timesteps / self.rarl_model.episode_length)
        self.update_freq_vanilla = self.vanilla_model.update_freq
        self.update_freq_rarl = self.rarl_model.update_freq
        self.n_vanilla_models = int(self.vanilla_model.model.num_timesteps / self.vanilla_model.n_steps)
        self.n_rarl_models = int(self.rarl_model.gov_model.num_timesteps / self.rarl_model.n_steps)
        self.vanilla_to_norm = os.path.exists(os.path.join(self.vanilla_dir, "envs/1.pkl"))
        self.rarl_to_norm = os.path.exists(os.path.join(self.rarl_dir, "envs/1.pkl"))
        self.vanilla_episode_checkpoint_multiplier = int(self.vanilla_model.n_steps / self.vanilla_model.episode_length)
        self.rarl_episode_checkpoint_multiplier = int(self.rarl_model.n_steps / self.rarl_model.episode_length)


    def add_shifted_env_range(
        self,
        shift_types = ["behavioural"],
        shift: dict = {},
        name: str = None
    ):
        
        if name == None:
            name = "shifted_env_range_" + str(len(self.shifted_envs_range) + 1)
        
        shifted_env_config = deepcopy(self.train_env_config)

        n_models = 1

        if "behavioural" in shift_types:

            # n_values
            n_values = shift["n_values"]
            n_models = shift["n_models"]
            # fip or cwr or mp
            param = shift["param"]

            low = shift["low"]
            high = shift["high"]

            values = [low + (iter * (high - low) / n_values) for iter in range(n_values + 1)]

            # self.shifted_envs_range[param] = {}

            self.shifted_envs_range[name] = (param, values)

        self._add_datasets_range(name, n_models)

        self._plot_for_range(name)

        self.save()

        # "shift" contains multiple params
        #   (as of now we assume shift_types = ["behavioural"])
        #   

    def _add_datasets_range(
        self,
        name,
        n_models
    ):
        shifted_env_config = deepcopy(self.train_env_config)

        shifted_env_config["additional"]["change"] = True

        param = self.shifted_envs_range[name][0]
        values = self.shifted_envs_range[name][1]

        vanilla_dataset = np.zeros(shape = (len(values), n_models), dtype = np.float32)

        rarl_dataset = np.zeros(shape = (len(values), n_models), dtype = np.float32)

        for value_iter in range(len(values)):

            value = values[value_iter]
            
            shifted_env_config["additional"]["params_to_change"] = {param: value}

            shifted_env = self.train_env_class(**shifted_env_config)

            vanilla_tester = ModelTester(
                algo = self.vanilla_model.algorithm, 
                dir = self.vanilla_dir, 
                norm = self.vanilla_to_norm, 
                test_env = shifted_env
            )

            rarl_tester = ModelTester(
                algo = self.rarl_model.gov_algorithm,
                dir = self.rarl_dir,
                norm = self.rarl_to_norm,
                test_env = shifted_env
            )

            # vanilla_dataset = np.zeros(shape = (n_models,), dtype = np.float32)

            for iter in range(n_models):

                vanilla_tester.load(
                    checkpoint_no = int(iter + self.n_vanilla_models - n_models + 1)
                )

                vanilla_dataset[value_iter][iter] = vanilla_tester.compute_reward(n_samples = self.n_samples)

                rarl_tester.load(
                    checkpoint_no = int(iter + self.n_rarl_models - n_models + 1)
                )

                rarl_dataset[value_iter][iter] = rarl_tester.compute_reward(n_samples = self.n_samples)
            
        self.datasets_range[f"vanilla_{name}"] = vanilla_dataset
        
        self.datasets_range[f"rarl_{name}"] = rarl_dataset


        with open(
            file = os.path.join(
                self.vanilla_dir, 
                f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{len(values)}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(vanilla_dataset.tolist(), file)

    
        with open(
            file = os.path.join(
                self.rarl_dir, 
                f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{len(values)}.json"
            ), 
            mode = "w"
        ) as file:
            json.dump(rarl_dataset.tolist(), file)


    def _plot_for_range(
        self,
        name
    ):
        dir = os.path.join(self.dir, "range_plots")
        os.makedirs(dir, exist_ok = True)
        
        values = self.shifted_envs_range[name][1]
        param = values = self.shifted_envs_range[name][0]
        
        vanilla_dataset = self.datasets_range[f"vanilla_{name}"]
        rarl_dataset = self.datasets_range[f"rarl_{name}"]

        vanilla_means = np.mean(vanilla_dataset, axis = 1)
        vanilla_std_devs = np.std(vanilla_dataset, axis = 1)

        rarl_means = np.mean(rarl_dataset, axis = 1)
        rarl_std_devs = np.std(rarl_dataset, axis = 1)

        plt.ioff()

        plt.errorbar(values, vanilla_means, yerr = vanilla_std_devs, fmt = "o", color = "green", ecolor = "green", capsize = 5)
        plt.plot(values, vanilla_means, color = "green", label = "Vanilla")

        plt.errorbar(values, rarl_means, yerr = rarl_std_devs, fmt = "o", color = "blue", ecolor = "blue", capsize = 5)
        plt.plot(values, rarl_means, color = "blue", label = "RARL")

        plt.xlabel(param)
        plt.ylabel("Rewards")

        plt.title(f"Rewards on {param} from {values[0]} to {values[-1]}.png")

        plt.legend()

        plt.savefig(os.path.join(dir, f"{name}_{values[0]}_to_{values[-1]}"))
        
        plt.close()

        

    def save(self):

        save_dir = os.path.join(self.dir, "saved")
        os.makedirs(save_dir, exist_ok = True)

        with open(os.path.join(save_dir, "evaluator_range.pkl"), "wb") as file:
            pickle.dump(self, file)
        
        file_path = os.path.join(save_dir, "evaluator_range.pkl")

        print(f"RobustnessEvaluation object saved to {file_path}")
        

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as file:
            obj = pickle.load(file)
        
        print(f"RobustnessEvaluation object loaded from {file_path}")
        
        return obj

