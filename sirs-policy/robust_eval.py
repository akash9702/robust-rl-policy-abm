from sirs_gov import SIRSGov
# from juliacall import Main as jl
from model import VanillaModelManager, ModelTester, RARLModelManager, RandModelManager, RARLModelManager2, ARLModelManager
from copy import deepcopy
import os
import json
import torch
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from kl_div import calc_kl
from mmd import calc_mmd




class EvaluateRobustness:

    # TODO: change some of these to make more legible
    param_to_label = {
        "consumption_wealth_ratio" : "Ratio of Consumption to Wealth",
        "firm_invest_prob" : "Probability of Firm Investment",
        "memory_parameter" : "Memory Parameter",
        "labour_prod" : "Productivity of Labour",
        "capital_prod" : "Productivity of Capital",
        "alpha" : "Infectivity Rate",
        "beta" : "Recovery Rate",
        "gamma" : "Re-susceptibility Rate"
    }

    param_to_name = {
        "cwr" : "Ratio of Consumption to Wealth",
        "fip" : "Probability of Firm Investment",
        "mp" : "Memory Parameter",
        "lp" : "Productivity of Labour",
        "cp" : "Productivity of Capital",
        "alpha" : "Infectivity Rate",
        "beta" : "Recovery Rate",
        "gamma" : "Re-susceptibility Rate"
    }

    div_to_label = {
        "kl" : "KL-Divergence",
        "mmd" : "MMD"
    }

    def __init__(
        self,
        dir: str = None,
        n_seeds: int = 10,
        n_train_seeds: int = 3,
        kl_div_M: int = 100
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

        self.fixed_shift_config = {}
        self.fixed_shift_datasets = {}
        self.kl_div_M = kl_div_M

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
                param
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
            # low = np.exp(low)
            # high = np.exp(high)
            values = np.logspace(np.log10(low), np.log10(high), n_values, dtype = np.float32)

        if "values" in shift:
            values = shift["values"]
            if not isinstance(values, np.ndarray):
                values = np.array(values)

        self.param_shifted_envs[name] = (param, values)

        self._add_datasets_param_shift_one(name)

        # self._plot_for_param_shift_one(name)

    
    def _add_datasets_param_shift_one(
        self,
        name
    ):
        # print(f"adding datasets for {name}")

        shifted_env_config = deepcopy(self.env_config)
        shifted_env_config["additional"]["change"] = True

        param, values = self.param_shifted_envs[name]
        
        # print(name)
        # print(param)
        # print(values)
        # print(f"self.seeds is {self.seeds}")

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


                shifted_env_config = deepcopy(self.env_config)
                shifted_env_config["additional"]["change"] = True

                shifted_env_config["additional"]["params_to_change"] = {
                    param: value
                }

                shifted_env_config["additional"][param] = torch.Tensor([value])

                for seed_iter in range(self.n_seeds):
                    
                    seed = self.seeds[seed_iter]

                    # if value_iter == 0:
                        # print(f"train_seed_iter: {train_seed_iter}, seed_iter: {seed_iter}, seed: {seed}")

                    shifted_env_config["seed"] = seed

                    shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                    if seed_iter == 0 and train_seed_iter == 0:
                        print(shifted_env_v.params)
                        print(shifted_env_a.params)
                        print(shifted_env_r.params)
                        print()

                    # print(value)
                    # print(f"alpha, beta, gamma = {shifted_env_v.params}")

                    vanilla_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "Vanilla"),
                        test_env = shifted_env_v,
                        norm = True,
                        seed = seed
                    )
                    vanilla_model_tester.load_last()

                    vanilla_dataset[train_seed_iter][value_iter][seed_iter] = vanilla_model_tester.compute_reward(n_samples = 1)

                    arl_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "ARL"),
                        test_env = shifted_env_a,
                        norm = True,
                        seed = seed
                    )
                    arl_model_tester.load_last()

                    arl_dataset[train_seed_iter][value_iter][seed_iter] = arl_model_tester.compute_reward(n_samples = 1)

                    rarl_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "RARL"),
                        test_env = shifted_env_r,
                        norm = True,
                        seed = seed
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
            json.dump(vanilla_dataset.tolist(), file, indent = 4)

        with open(file = os.path.join(this_dir, f"arl_rewards.json"), mode = "w") as file:
            json.dump(arl_dataset.tolist(), file, indent = 4)

        with open(file = os.path.join(this_dir, f"rarl_rewards.json"), mode = "w") as file:
            json.dump(rarl_dataset.tolist(), file, indent = 4)

        

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
        plt.ioff()

        plot_dir = os.path.join(self.plot_dir, name)
        os.makedirs(plot_dir, exist_ok = True)

        param, values = self.param_shifted_envs[name]

        # values = np.exp(values)

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


        plt.rcParams.update({'font.size': 12})  # Set the global font size to 12

        plt.rcParams.update({
            'axes.labelsize': 14,    # Font size for x and y labels
            'xtick.labelsize': 16,   # Font size for x-axis tick labels
            'ytick.labelsize': 16,   # Font size for y-axis tick labels
            'axes.titlesize': 12     # Font size for the subplot titles
        })

        # Create the 1x3 subplot layout
        # fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))

        plt.ioff()

        # plt.rcParams.update({'font.size': 8})  # Set the global font size to 12

        # plt.rcParams.update({
        #     'axes.labelsize': 8,    # Font size for x and y labels
        #     'xtick.labelsize': 8,   # Font size for x-axis tick labels
        #     'ytick.labelsize': 8,   # Font size for y-axis tick labels
        #     'axes.titlesize': 8     # Font size for the subplot titles
        # })

        values = np.exp(values)

        axs[0].plot(values, vanilla_means, color = "green", label = "Vanilla")
        axs[0].fill_between(values, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[0].plot(values, arl_means, color = "blue", label = "Episode-Adv")
        axs[0].fill_between(values, arl_min, arl_max, color = "blue", alpha = 0.5)

        axs[0].set_xlabel(self.param_to_label[param])
        axs[0].set_ylabel("Rewards")

        axs[0].margins(0.2)  # Increase margins

        axs[0].set_xscale('log')

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
        axs[1].set_ylabel("Rewards")

        axs[1].margins(0.2)  # Increase margins

        axs[1].legend()

        axs[1].set_xscale('log')

        y_lim_min, y_lim_max = axs[1].get_ylim()

        y_min = np.minimum(vanilla_min.min(), rarl_min.min())
        y_max = np.maximum(vanilla_max.max(), rarl_max.max())

        axs[1].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )

        # plt.title(self.param_to_label[param])

        plt.tight_layout()

        plt.savefig(f"{plot_dir}/vanilla_vs_rarl_vs_arl.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/fixed-one", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/fixed-one/{name}.png", dpi = 300)

        os.makedirs(f"{self.plot_dir}/fixed-one-2", exist_ok=True)
        plt.savefig(f"{self.plot_dir}/fixed-one-2/{name}.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/fixed-one-2", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/fixed-one-2/{name}.png", dpi = 300)
        

        plt.close()

        

    def plot_existing_datasets_one_shift(
        self,
        name,
        shift
    ):
        
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

        values = np.exp(values)

        axs[0].set_xscale('log')
        axs[1].set_xscale('log')

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
        plt.savefig(f"{plot_dir}/{name}.png", dpi = 300)

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


        for train_seed_iter in range(self.n_train_seeds):

            shifted_env_config["additional"]["params_to_change"] = {}

            
            
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(self.dir, str(train_seed))

            for value_1_iter in range(n_values_1):

                value_1 = values_1[value_1_iter]


                shifted_env_config = deepcopy(self.env_config)

                shifted_env_config["additional"]["change"] = True

                # shifted_env_config["additional"]["params_to_change"][param_1] = value_1

                shifted_env_config["additional"][param_1] = torch.Tensor([value_1])

                

                for value_2_iter in range(n_values_2):

                    value_2 = values_2[value_2_iter]

                    shifted_env_config["additional"]["params_to_change"][param_2] = value_2

                    shifted_env_config["additional"][param_2] = torch.Tensor([value_2])

                    for seed_iter in range(n_seeds):

                        seed = self.seeds[seed_iter]

                        shifted_env_config["seed"] = seed

                        shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                        shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                        shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                        # if train_seed_iter == 0 and seed_iter == 0:
                            
                        #     print(f"params are {shifted_env_v.params}")

                        vanilla_model_tester = ModelTester(
                            algo = "PPO",
                            dir = os.path.join(curr_dir, "Vanilla"),
                            test_env = shifted_env_v,
                            norm = True,
                        )
                        vanilla_model_tester.load_last()

                        vanilla_dataset[train_seed_iter][value_1_iter][value_2_iter][seed_iter] = vanilla_model_tester.compute_reward(n_samples = 1)

                        arl_model_tester = ModelTester(
                            algo = "PPO",
                            dir = os.path.join(curr_dir, "ARL"),
                            test_env = shifted_env_a,
                            norm = True,
                        )
                        arl_model_tester.load_last()

                        arl_dataset[train_seed_iter][value_1_iter][value_2_iter][seed_iter] = arl_model_tester.compute_reward(n_samples = 1)

                        rarl_model_tester = ModelTester(
                            algo = "PPO",
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

        self.two_plot_name = name

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

            self._plot_2d(values, vanilla_dataset[train_seed_iter], arl_dataset[train_seed_iter], rarl_dataset[train_seed_iter], curr_dir, param_1, param_2)

        vanilla_dataset_mean = np.mean(vanilla_dataset, axis = 0)
        arl_dataset_mean = np.mean(arl_dataset, axis = 0)
        rarl_dataset_mean = np.mean(rarl_dataset, axis = 0)

        # self._plot_2d(values, vanilla_dataset_mean, "Vanilla", plot_dir, param_1, param_2)
        # self._plot_2d(values, arl_dataset_mean, "Episode-Adv", plot_dir, param_1, param_2)
        # self._plot_2d(values, rarl_dataset_mean, "Step-Adv", plot_dir, param_1, param_2)
        self._plot_2d(values, vanilla_dataset_mean, arl_dataset_mean, rarl_dataset_mean, plot_dir, param_1, param_2)

    def _plot_2d_2(
        self,
        values,
        vanilla_dataset,
        arl_dataset,
        rarl_dataset,
        plot_dir,
        param_1,
        param_2
    ):
        plt.ioff()

        os.makedirs(plot_dir, exist_ok=True)

        X = values
        Y1 = vanilla_dataset
        Y2 = arl_dataset
        Y3 = rarl_dataset

        x_coords = [coord[0] for row in X for coord in row]
        y_coords = [coord[1] for row in X for coord in row]

        y1_values = [value for row in Y1 for value in row]
        y2_values = [value for row in Y2 for value in row]
        y3_values = [value for row in Y3 for value in row]

        global_min = np.min([np.min(y1_values), np.min(y2_values), np.min(y3_values)])
        global_max = np.max([np.max(y1_values), np.max(y2_values), np.max(y3_values)])

        norm = plt.Normalize(global_min, global_max)
        cmap = plt.cm.coolwarm

        plt.rcParams.update({'font.size': 10})

        fig, axs = plt.subplots(1, 3, figsize=(9, 3))

        x_coords = np.exp(x_coords)
        y_coords = np.exp(y_coords)

        # First heatmap (Y1)
        heatmap1 = axs[0].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y1_values, cmap=cmap, norm=norm)
        axs[0].set_xlabel(self.param_to_label[param_1])
        axs[0].set_ylabel(self.param_to_label[param_2])
        axs[0].set_title("Vanilla")

        # Second heatmap (Y2)
        heatmap2 = axs[1].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y2_values, cmap=cmap, norm=norm)
        axs[1].set_xlabel(self.param_to_label[param_1])
        axs[1].set_title("Episode-Adv")

        # Third heatmap (Y3)
        heatmap3 = axs[2].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y3_values, cmap=cmap, norm=norm)
        axs[2].set_xlabel(self.param_to_label[param_1])
        axs[2].set_title("Step-Adv")

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='Reward Spectrum')

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-1", exist_ok=True)
        plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-1/{self.two_plot_name}.png", dpi=300)

        plt.close()


    def _plot_2d(
        self,
        values,
        vanilla_dataset,
        arl_dataset,
        rarl_dataset,
        # dataset_name,
        plot_dir,
        param_1,
        param_2
    ):
        plt.ioff()

        os.makedirs(plot_dir, exist_ok = True)

        X = values
        Y1 = vanilla_dataset
        Y2 = arl_dataset
        Y3 = rarl_dataset

        # print('PRINTING VALUES ARRAY SHAPE')
        # print(np.array(values).shape)
        # print(len(values[0]))
        # print(len(values[:,0]))

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


        # values = np.exp(values)

        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[2].set_xscale('log')

        axs[0].set_yscale('log')
        axs[1].set_yscale('log')
        axs[2].set_yscale('log')

        x_coords = np.exp(x_coords)
        y_coords = np.exp(y_coords)

        # First plot (Y1)
        scatter1 = axs[0].scatter(x_coords, y_coords, c=y1_values, cmap=cmap, norm=norm, s=100)
        # heatmap1 = axs[0].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y1_values, cmap=cmap, norm=norm)
        
        axs[0].set_xlabel(self.param_to_label[param_1])
        axs[0].set_ylabel(self.param_to_label[param_2])
        axs[0].margins(0.2)  # Increase margins
        axs[0].set_title("Vanilla")

        # Second plot (Y2)
        scatter2 = axs[1].scatter(x_coords, y_coords, c=y2_values, cmap=cmap, norm=norm, s=100)
        # heatmap2 = axs[1].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y2_values, cmap=cmap, norm=norm)
        
        axs[1].set_xlabel(self.param_to_label[param_1])
        # axs[1].set_ylabel(self.param_to_label[param_2])
        axs[1].margins(0.2)  # Increase margins
        axs[1].set_title("Episode-Adv")

        # Third plot (Y3)
        scatter3 = axs[2].scatter(x_coords, y_coords, c=y3_values, cmap=cmap, norm=norm, s=100)
        # heatmap3 = axs[2].hist2d(x_coords, y_coords, bins=(len(values[0]), len(values[:,0])), weights=y3_values, cmap=cmap, norm=norm)
        
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
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two/{self.two_plot_name}.png", dpi = 300)
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-2", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-1/{self.two_plot_name}.png", dpi = 300)
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-2", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-2/{self.two_plot_name}.png", dpi = 300)
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-3", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-3/{self.two_plot_name}.png", dpi = 300)
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-4", exist_ok=True)
        # plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-4/{self.two_plot_name}.png", dpi = 300)
        os.makedirs(f"{self.dir}/robustness_comparison_plots/fixed-two-5", exist_ok=True)
        plt.savefig(f"{self.dir}/robustness_comparison_plots/fixed-two-5/{self.two_plot_name}.png", dpi = 300)

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

        # for train_seed_iter in range(self.n_train_seeds):
        #     train_seed = self.train_seeds[train_seed_iter]

        #     curr_dir = os.path.join(plot_dir, str(train_seed))

        #     # self._plot_2d(values, vanilla_dataset[train_seed_iter], "Vanilla", curr_dir, param_1, param_2)
        #     # self._plot_2d(values, arl_dataset[train_seed_iter], "Episode-Adv", curr_dir, param_1, param_2)
        #     # self._plot_2d(values, rarl_dataset[train_seed_iter], "Step-Adv", curr_dir, param_1, param_2)

        #     self._plot_2d(values, vanilla_dataset[train_seed_iter], arl_dataset[train_seed_iter], rarl_dataset[train_seed_iter], curr_dir, param_1, param_2)

        vanilla_dataset_mean = np.mean(vanilla_dataset, axis = 0)
        arl_dataset_mean = np.mean(arl_dataset, axis = 0)
        rarl_dataset_mean = np.mean(rarl_dataset, axis = 0)

        # print(vanilla_dataset_mean)
        # print(arl_dataset_mean)

        # self._plot_2d(values, vanilla_dataset_mean, "Vanilla", plot_dir, param_1, param_2)
        # self._plot_2d(values, arl_dataset_mean, "Episode-Adv", plot_dir, param_1, param_2)
        # self._plot_2d(values, rarl_dataset_mean, "Step-Adv", plot_dir, param_1, param_2)
        self._plot_2d(values, vanilla_dataset_mean, arl_dataset_mean, rarl_dataset_mean, plot_dir, param_1, param_2)
        
    

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
                    algo = "PPO",
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
                    algo = "PPO",
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
                    algo = "PPO",
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

        for i in range(1, len(parts)):
            # "{:.2g}".format(num1)
            parts[i] = str("{:.2g}".format(np.exp(float(parts[i]))))
        
        # The rest are the range values
        range_values = "-".join(parts[1:])
        
        # Format the output string using param_to_name dictionary
        return f"{self.param_to_name[param]}, {range_values}"
        
    def _plot_adv(
        self,
        name
    ):
        plot_dir = os.path.join(self.plot_dir, "adv2")
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

        # plt.xlim(-0.5, len(x) - 0.5)
        # plt.ylim(min(mins) - 5, max(maxs) + 5)

        plot_title = self.format_param_string(param_string=name)

        plt.title(plot_title, fontsize = 13)

        plt.tight_layout()

        plt.savefig(f"{plot_dir}/{name}.png", dpi = 300)
        
        plt.close()



    # DISTRIBUTIONAL SHIFT FORMULATION

    def add_fixed_shift(
        self,
        name,
        shift
    ):
        """
        plots performance for fixed parameter shifts
        along with each plot, there should be a text file describing the exact kind of shift

        args:
            shift: dict that contains
                params_to_shift: list of parameters that should be shifted
                std_devs: list of standard deviations from which shifted parameters should be samples
                    e.g. if params = ["alpha", "beta"] and std_devs = [2.0, 1.0], then for each shifted environment, alpha would be sampled from N(original_alpha_value = 1.5, alpha_std_dev = 2.0) and beta from N(-2.0, 1.0)
                n_shifts: number of shifted environments to be sampled for this shift. default value should be 20.
            name: name of the shift
                type: type of sampling: uniform, random, values given?
                e.g. if params = ["alpha", "beta"] and std_devs = [2.0, 1.0], then name = "alpha_2.0_beta_1.0"

        notes:
            > ideally, it shouldn't matter what the "type" of shift is -- since we are thinking of all shifts in terms of distributional shifts, we should only have one graph in which we simply look at many different dist shifts -- each of which has all the possible parameters shifted 
                > one rebuttal to this is that having all the possible parameters shifted could generally lead to very large shifts -- this is especially true in M-ABM, even though it might not be true for SIRS
                > hence, we might need to have many different plots -- one for each possible subset of parameters to shift
            
        """

        original_params = deepcopy(self.env_config["base_params"])

        if "params_to_shift" not in shift:
            raise ValueError("shift needs to contain key params_to_shift")
        params_to_shift = shift["params_to_shift"]

        if "type_of_values" not in shift:
            shift["type_of_values"] = "random"
        type_of_values = shift["type_of_values"]

        if type_of_values == "uniform":

            if "lows" not in shift:
                shift["lows"] = [
                    original_params[0] - float("alpha" in shift["params_to_shift"]) * 2.0,
                    original_params[0] - float("beta" in shift["params_to_shift"]) * 2.0,
                    original_params[0] - float("gamma" in shift["params_to_shift"]) * 2.0
                ]
            lows = shift["lows"]
            
            if "highs" not in shift:
                shift["highs"] = [
                    original_params[0] + float("alpha" in shift["params_to_shift"]) * 2.0,
                    original_params[0] + float("beta" in shift["params_to_shift"]) * 2.0,
                    original_params[0] + float("gamma" in shift["params_to_shift"]) * 2.0
                ]
            highs = shift["highs"]

            if "n_shifts_list" not in shift:
                shift["n_shifts"] = [
                    7 if "alpha" in shift["params_to_shift"] else 1,
                    7 if "beta" in shift["params_to_shift"] else 1,
                    7 if "gamma" in shift["params_to_shift"] else 1
                    # int("alpha" in shift["params_to_shift"]) * 7,
                    # int("beta" in shift["params_to_shift"]) * 7,
                    # int("gamma" in shift["params_to_shift"]) * 7
                ]
            n_shifts_list = shift["n_shifts_list"]
            n_shifts = n_shifts_list[0] * n_shifts_list[1] * n_shifts_list[2]

            shift_configs = []

            for i in range(n_shifts_list[0]):
                shifted_alpha = original_params[0] if "alpha" not in shift["params_to_shift"] else lows[0] + (highs[0] - lows[0]) * i / (n_shifts_list[0] - 1)
                for j in range(n_shifts_list[1]):
                    shifted_beta = original_params[1] if "beta" not in shift["params_to_shift"] else lows[1] + (highs[1] - lows[1]) * j / (n_shifts_list[1] - 1)
                    for k in range(n_shifts_list[2]):
                        shifted_gamma = original_params[2] if "gamma" not in shift["params_to_shift"] else lows[2] + (highs[2] - lows[2]) * k / (n_shifts_list[2] - 1)

                        shift_configs.append([shifted_alpha, shifted_beta, shifted_gamma])
            
            shift_configs = np.array(shift_configs)

            

        elif type_of_values == "values":

            if "values" not in shift:
                raise ValueError("if type_of_values == \"values\" then shift[\"values\"] should be provided")
            # assumed that values = an array which contains elements that are shifted_params
            shift_configs = np.array(shift["values"])

        elif type_of_values == "random":

            if "std_devs" not in shift:
                shift["std_devs"] = [2.0, 2.0, 2.0]
            std_devs = np.array(shift["std_devs"])

            if "n_shifts" not in shift:
                shift["n_shifts"] = 10
            n_shifts = shift["n_shifts"]

            shift_configs = []

            for i in range(n_shifts):
                shifted_params = deepcopy(original_params)
                # if "alpha" in params_to_shift:
                #     shifted_params[0] = np.random.normal(loc = original_params[0], scale = std_devs[j])
                # if "beta" in params_to_shift:
                #     shifted_params[1] = np.random.normal(loc = original_params[1], scale = std_devs[j])
                # if "gamma" in params_to_shift:
                #     shifted_params[2] = np.random.normal(loc = original_params[2], scale = std_devs[j])
                for j in range(len(params_to_shift)):
                    param = params_to_shift[j]
                    if param == "alpha":
                        shifted_params[0] = np.random.normal(loc = original_params[0], scale = std_devs[j])
                    elif param == "beta":
                        shifted_params[1] = np.random.normal(loc = original_params[1], scale = std_devs[j])
                    elif param == "gamma":
                        shifted_params[2] = np.random.normal(loc = original_params[2], scale = std_devs[j])
                shift_configs.append(shifted_params)

            shift_configs = np.array(shift_configs)

        type_of_divergence = "mmd"
        if "type_of_divergence" in shift:
            type_of_divergence = shift["type_of_divergence"]


        # shift_configs = []

        # original_params = deepcopy(self.env_config["base_params"])

        # for i in range(n_shifts):
        #     shifted_params = deepcopy(original_params)
        #     # if "alpha" in params_to_shift:
        #     #     shifted_params[0] = np.random.normal(loc = original_params[0], scale = std_devs[j])
        #     # if "beta" in params_to_shift:
        #     #     shifted_params[1] = np.random.normal(loc = original_params[1], scale = std_devs[j])
        #     # if "gamma" in params_to_shift:
        #     #     shifted_params[2] = np.random.normal(loc = original_params[2], scale = std_devs[j])
        #     for j in range(len(params_to_shift)):
        #         param = params_to_shift[j]
        #         if param == "alpha":
        #             shifted_params[0] = np.random.normal(loc = original_params[0], scale = std_devs[j])
        #         elif param == "beta":
        #             shifted_params[1] = np.random.normal(loc = original_params[1], scale = std_devs[j])
        #         elif param == "gamma":
        #             shifted_params[2] = np.random.normal(loc = original_params[2], scale = std_devs[j])
        #     shift_configs.append(shifted_params)


        for shifted_params in shift_configs:
            print(shifted_params)


        self.fixed_shift_config[name] = (shift_configs, params_to_shift, n_shifts, type_of_divergence)

        self.add_fixed_shift_datasets(name)
    
        self.plot_fixed_shift(name)

    def add_fixed_shift_datasets(
        self,
        name
    ):
        shift_configs, params_to_shift, n_shifts, type_of_divergence  = self.fixed_shift_config[name]

        vanilla_dataset = np.zeros(shape = (self.n_train_seeds, n_shifts, self.n_seeds), dtype = np.float32)
        arl_dataset = np.zeros(shape = (self.n_train_seeds, n_shifts, self.n_seeds), dtype = np.float32)
        rarl_dataset = np.zeros(shape = (self.n_train_seeds, n_shifts, self.n_seeds), dtype = np.float32)


        for train_seed_iter in range(self.n_train_seeds):
            
            train_seed = self.train_seeds[train_seed_iter]

            curr_dir = os.path.join(self.dir, str(train_seed))

            for shift_iter in range(n_shifts):

                shifted_params = shift_configs[shift_iter]

                shifted_env_config = deepcopy(self.env_config)

                shifted_env_config["additional"]["change"] = True

                # shifted_env_config["additional"]["params_to_change"] = {}

                # shifted_env_config["additional"]["params_to_change"]["alpha"] = shifted_params[0]
                # shifted_env_config["additional"]["params_to_change"]["beta"] = shifted_params[1]
                # shifted_env_config["additional"]["params_to_change"]["gamma"] = shifted_params[2]

                shifted_env_config["additional"]["alpha"] = torch.tensor([shifted_params[0]])
                shifted_env_config["additional"]["beta"] = torch.tensor([shifted_params[1]])
                shifted_env_config["additional"]["gamma"] = torch.tensor([shifted_params[2]])

                for seed_iter in range(self.n_seeds):

                    seed = self.seeds[seed_iter]

                    shifted_env_config["seed"] = seed

                    shifted_env_v = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_a = self.env_class(**deepcopy(shifted_env_config))
                    shifted_env_r = self.env_class(**deepcopy(shifted_env_config))

                    if seed_iter == 0 and train_seed_iter == 0:
                        print(shifted_env_v.params)
                        print(shifted_env_a.params)
                        print(shifted_env_r.params)
                        print()

                    # print(value)
                    # print(f"alpha, beta, gamma = {shifted_env_v.params}")

                    vanilla_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "Vanilla"),
                        test_env = shifted_env_v,
                        norm = True,
                        seed = seed
                    )
                    vanilla_model_tester.load_last()

                    vanilla_dataset[train_seed_iter][shift_iter][seed_iter] = vanilla_model_tester.compute_reward(n_samples = 1)

                    arl_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "ARL"),
                        test_env = shifted_env_a,
                        norm = True,
                        seed = seed
                    )
                    arl_model_tester.load_last()

                    arl_dataset[train_seed_iter][shift_iter][seed_iter] = arl_model_tester.compute_reward(n_samples = 1)

                    rarl_model_tester = ModelTester(
                        algo = "PPO",
                        dir = os.path.join(curr_dir, "RARL"),
                        test_env = shifted_env_r,
                        norm = True,
                        seed = seed
                    )
                    rarl_model_tester.load_last()

                    rarl_dataset[train_seed_iter][shift_iter][seed_iter] = rarl_model_tester.compute_reward(n_samples = 1)

        self.fixed_shift_datasets[f"vanilla_{name}"] = vanilla_dataset
        self.fixed_shift_datasets[f"arl_{name}"] = arl_dataset
        self.fixed_shift_datasets[f"rarl_{name}"] = rarl_dataset

        self.fixed_datasets_dir = os.path.join(self.dir, f"datasets/{name}")
        os.makedirs(self.fixed_datasets_dir, exist_ok = True)

        with open(file = os.path.join(self.fixed_datasets_dir, f"vanilla_rewards.json"), mode = "w") as file:
            json.dump(vanilla_dataset.tolist(), file, indent = 4)

        with open(file = os.path.join(self.fixed_datasets_dir, f"arl_rewards.json"), mode = "w") as file:
            json.dump(arl_dataset.tolist(), file, indent = 4)

        with open(file = os.path.join(self.fixed_datasets_dir, f"rarl_rewards.json"), mode = "w") as file:
            json.dump(rarl_dataset.tolist(), file, indent = 4)


    def plot_fixed_shift(
        self,
        name
    ):
        
        shift_configs, params_to_shift, n_shifts, type_of_divergence  = self.fixed_shift_config[name]

        vanilla_dataset = self.fixed_shift_datasets[f"vanilla_{name}"]
        arl_dataset = self.fixed_shift_datasets[f"arl_{name}"]
        rarl_dataset = self.fixed_shift_datasets[f"rarl_{name}"]


        if type_of_divergence == "kl":
            divergence_func = calc_kl
        elif type_of_divergence == "mmd":
            divergence_func = calc_mmd

        divs = np.zeros(shape = (n_shifts, ), dtype = np.float32)

        for shift_iter in range(n_shifts):

            divs[shift_iter] = divergence_func(
                M = self.kl_div_M,
                N = self.env_config["N"],
                initial_infected = self.env_config["initial_infected"],
                original_params = deepcopy(np.array(self.env_config["base_params"])),
                shifted_params = deepcopy(shift_configs[shift_iter]),
                T = self.env_config["T"]
            )
        
        divs_unsorted = deepcopy(divs)

        vanilla_dataset = np.mean(vanilla_dataset, axis = 2)
        arl_dataset = np.mean(arl_dataset, axis = 2)
        rarl_dataset = np.mean(rarl_dataset, axis = 2)

        for train_seed_iter in range(self.n_train_seeds):

            this_divs_unsorted = deepcopy(divs_unsorted)
            this_vanilla = deepcopy(vanilla_dataset[train_seed_iter]).tolist()

            this_divs_unsorted, this_vanilla = zip(*sorted(zip(this_divs_unsorted, this_vanilla)))
            vanilla_dataset[train_seed_iter] = this_vanilla

            this_divs_unsorted = deepcopy(divs_unsorted)
            this_arl = deepcopy(arl_dataset[train_seed_iter]).tolist()

            this_divs_unsorted, this_arl = zip(*sorted(zip(this_divs_unsorted, this_arl)))
            arl_dataset[train_seed_iter] = this_arl

            this_divs_unsorted = deepcopy(divs_unsorted)
            this_rarl = deepcopy(rarl_dataset[train_seed_iter]).tolist()

            this_divs_unsorted, this_rarl = zip(*sorted(zip(this_divs_unsorted, this_rarl)))
            rarl_dataset[train_seed_iter] = this_rarl

        divs = sorted(divs)
        





        # kl_divs = np.zeros(shape = (n_shifts,), dtype = np.float32)

        # for shift_iter in range(n_shifts):
            
        #     kl_divs[shift_iter] = calc_kl(
        #         M = self.kl_div_M,
        #         N = self.env_config["N"],
        #         initial_infected = self.env_config["initial_infected"],
        #         original_params = deepcopy(np.array(self.env_config["base_params"])),
        #         shifted_params = deepcopy(shift_configs[shift_iter]),
        #         T = self.env_config["T"]
        #     )
        
        # kl_divs_unsorted = deepcopy(kl_divs)

        # vanilla_dataset = np.mean(vanilla_dataset, axis = 2)
        # arl_dataset = np.mean(arl_dataset, axis = 2)
        # rarl_dataset = np.mean(rarl_dataset, axis = 2)

        # for train_seed_iter in range(self.n_train_seeds):
            
        #     this_kl_unsorted = deepcopy(kl_divs_unsorted)
        #     this_vanilla = deepcopy(vanilla_dataset[train_seed_iter]).tolist()

        #     this_kl_unsorted, this_vanilla = zip(*sorted(zip(this_kl_unsorted, this_vanilla)))
        #     vanilla_dataset[train_seed_iter] = this_vanilla

        #     this_kl_unsorted = deepcopy(kl_divs_unsorted)
        #     this_arl = deepcopy(arl_dataset[train_seed_iter]).tolist()

        #     this_kl_unsorted, this_arl = zip(*sorted(zip(this_kl_unsorted, this_arl)))
        #     arl_dataset[train_seed_iter] = this_arl

        #     this_kl_unsorted = deepcopy(kl_divs_unsorted)
        #     this_rarl = deepcopy(rarl_dataset[train_seed_iter]).tolist()

        #     this_kl_unsorted, this_rarl = zip(*sorted(zip(this_kl_unsorted, this_rarl)))
        #     rarl_dataset[train_seed_iter] = this_rarl

        # kl_divs = sorted(kl_divs)

        

        # PLOTTING WITH FILL BETWEEN

        vanilla_means = np.mean(vanilla_dataset, axis = 0)
        arl_means = np.mean(arl_dataset, axis = 0)
        rarl_means = np.mean(rarl_dataset, axis = 0)

        vanilla_min = np.min(vanilla_dataset, axis = 0)
        arl_min = np.min(arl_dataset, axis = 0)
        rarl_min = np.min(rarl_dataset, axis = 0)

        vanilla_max = np.max(vanilla_dataset, axis = 0)
        arl_max = np.max(arl_dataset, axis = 0)
        rarl_max = np.max(rarl_dataset, axis = 0)


        plt.ioff()


        # plt.rcParams.update({'font.size': 12})  # Set the global font size to 12

        # plt.rcParams.update({
        #     'axes.labelsize': 14,    # Font size for x and y labels
        #     'xtick.labelsize': 16,   # Font size for x-axis tick labels
        #     'ytick.labelsize': 16,   # Font size for y-axis tick labels
        #     'axes.titlesize': 12     # Font size for the subplot titles
        # })

        

        # Create the 1x3 subplot layout
        # fig, axs = plt.subplots(1, 3, figsize=(9, 3))
        fig, axs = plt.subplots(1, 2, figsize=(9, 3))



        axs[0].plot(divs, vanilla_means, color = "green", label = "Vanilla")
        axs[0].fill_between(divs, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[0].plot(divs, arl_means, color = "blue", label = "Episode-Adv")
        axs[0].fill_between(divs, arl_min, arl_max, color = "blue", alpha = 0.5)

        axs[0].set_xlabel(self.div_to_label[type_of_divergence])
        axs[0].set_ylabel("Rewards")

        axs[0].margins(0.2)  # Increase margins

        # axs[0].set_xscale('log')

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

        axs[1].plot(divs, vanilla_means, color = "green", label = "Vanilla")
        axs[1].fill_between(divs, vanilla_min, vanilla_max, color = "green", alpha = 0.5)

        axs[1].plot(divs, rarl_means, color = "blue", label = "Step-Adv")
        axs[1].fill_between(divs, rarl_min, rarl_max, color = "blue", alpha = 0.5)

        axs[1].set_xlabel(self.div_to_label[type_of_divergence])
        axs[1].set_ylabel("Rewards")

        axs[1].margins(0.2)  # Increase margins

        axs[1].legend()

        # axs[1].set_xscale('log')

        y_lim_min, y_lim_max = axs[1].get_ylim()

        y_min = np.minimum(vanilla_min.min(), rarl_min.min())
        y_max = np.maximum(vanilla_max.max(), rarl_max.max())

        axs[1].set_ylim(
            [
                y_lim_min,
                y_lim_max + 0.5 * (y_max - y_min)
            ]
        )

        # plt.title(self.param_to_label[param])

        plt.tight_layout()

        plot_dir = os.path.join(self.plot_dir, f"fixed-shifts/{name}")
        os.makedirs(plot_dir, exist_ok=True)

        plt.savefig(f"{plot_dir}/vanilla_vs_rarl_vs_arl.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/fixed-one", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/fixed-one/{name}.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/fixed-one-2", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/fixed-one-2/{name}.png", dpi = 300)

        # os.makedirs(f"{self.plot_dir}/fixed-one-2", exist_ok=True)
        # plt.savefig(f"{self.plot_dir}/fixed-one-2/{name}.png", dpi = 300)
        

        plt.close()

        param_distances = np.zeros(shape = (n_shifts,), dtype = np.float32)

        for shift_iter in range(n_shifts):
            this_shifted_params = shift_configs[shift_iter]

            original_params = deepcopy(self.env_config["base_params"])

            param_distances[shift_iter] = np.linalg.norm(np.array(this_shifted_params - original_params))
            param_distances[shift_iter] = np.array(this_shifted_params - original_params).sum()

        divs_to_plot = deepcopy(divs_unsorted)

        # for i in range(n_shifts):
        #     print(f"{param_distances[i]}, {divs_to_plot[i]}")

        param_distances, divs_to_plot = zip(*sorted(zip(param_distances, divs_to_plot)))

        # print()
        # for i in range(n_shifts):
        #     print(f"{param_distances[i]}, {divs_to_plot[i]}")

        plt.ion()
        plt.plot(param_distances, divs_to_plot)
        plt.show()
        plt.ioff()

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
                # print(os.path.join(this_vanilla_dir, "saved/model_config.json"))
                this_vanilla_model_config = json.load(file)
            
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

            self.rarl_models.append(RARLModelManager(**this_rarl_model_config))
            self.rarl_models[seed_iter].load()
        
        





class RobustnessEvaluationRange:

    def __init__(
        self,
        dir: str = None,
        n_samples: int = 1,
        which: list[bool] = [True, True, True, True],
        N_mu: int = 20,
        seeds: list[int] = [42]
    ):
        
        if dir == None:
            raise ValueError("dir cannot be None")
        self.dir = dir

        self.n_samples = n_samples

        self.seeds = seeds

        # which types of models have we trained in dir
        self.is_vanilla, self.is_rarl, self.is_rand, self.is_rarl2 = which
        if not self.is_vanilla:
            raise ValueError("there needs to be a Vanilla model in dir i.e. which[0] needs to be True")

        # dir variables
        
        self.plot_dir = os.path.join(self.dir, "range_plots")
        os.makedirs(self.plot_dir, exist_ok = True)

        if self.is_vanilla:
            
            # creating Vanilla model
            vanilla_dir = os.path.join(self.dir, "Vanilla")
            with open(os.path.join(vanilla_dir, "saved/model_config.json"), "r") as file:
                vanilla_model_config = json.load(file)
            self.vanilla_model = VanillaModelManager(**vanilla_model_config)
            # self.vanilla_model.load(os.path.join(vanilla_dir, "saved"))
            self.vanilla_model.load()

            self.vanilla_dir = os.path.join(self.dir, "Vanilla")

            self.vanilla_model_dir = os.path.join(self.vanilla_dir, "models")


            # model variables for plotting
            self.n_eps_vanilla = int(self.vanilla_model.model.num_timesteps / self.vanilla_model.episode_length)
            self.update_freq_vanilla = self.vanilla_model.update_freq
            self.n_vanilla_models = int(self.vanilla_model.model.num_timesteps / (self.vanilla_model.episode_length * N_mu * self.update_freq_vanilla))
            self.vanilla_episode_checkpoint_multiplier = int(self.vanilla_model.n_steps / self.vanilla_model.episode_length)
            # print(f"n_vanilla_models: {self.n_vanilla_models}")
        if self.is_rarl:
            # creating RARL model
            rarl_dir = os.path.join(self.dir, "RARL")
            with open(os.path.join(rarl_dir, "saved/model_config.json"), "r") as file:
                rarl_model_config = json.load(file)
            self.rarl_model = RARLModelManager(**rarl_model_config)
            # self.rarl_model.load(os.path.join(rarl_dir, "saved"))
            self.rarl_model.load()

            self.rarl_dir = os.path.join(self.dir, "RARL")

            self.rarl_model_dir = os.path.join(self.rarl_dir, "models")


            # model variables for plotting
            self.n_eps_rarl = int(self.rarl_model.gov_model.num_timesteps / self.rarl_model.episode_length)
            self.update_freq_rarl = self.rarl_model.update_freq
            self.n_rarl_models = int(self.rarl_model.gov_model.num_timesteps / (self.rarl_model.episode_length * N_mu * self.update_freq_rarl))
            self.rarl_episode_checkpoint_multiplier = int(self.rarl_model.n_steps / self.rarl_model.episode_length)
            # print(f"n_rarl_models: {self.n_rarl_models}")

        if self.is_rarl2:
            # creating RARL model
            rarl2_dir = os.path.join(self.dir, "RARL2")
            with open(os.path.join(rarl2_dir, "saved/model_config.json"), "r") as file:
                rarl2_model_config = json.load(file)
            self.rarl2_model = RARLModelManager2(**rarl2_model_config)
            # self.rarl2_model.load(os.path.join(rarl2_dir, "saved"))
            self.rarl2_model.load()

            self.rarl2_dir = os.path.join(self.dir, "RARL2")

            self.rarl2_model_dir = os.path.join(self.rarl2_dir, "models")


            # model variables for plotting
            self.n_eps_rarl2 = int(self.rarl2_model.gov_model.num_timesteps / self.rarl2_model.episode_length)
            self.update_freq_rarl2 = self.rarl2_model.update_freq
            self.n_rarl2_models = int(self.rarl2_model.gov_model.num_timesteps / (self.rarl2_model.episode_length * N_mu * self.update_freq_rarl2))
            self.rarl2_episode_checkpoint_multiplier = int(self.rarl2_model.n_steps / self.rarl2_model.episode_length)
            # print(f"n_rarl2_models: {self.n_rarl2_models}")
        if self.is_rand:

            # creating Vanilla model
            rand_dir = os.path.join(self.dir, "Rand")
            with open(os.path.join(vanilla_dir, "saved/model_config.json"), "r") as file:
                rand_model_config = json.load(file)
            self.rand_model = RandModelManager(**rand_model_config)
            # self.rand_model.load(os.path.join(rand_dir, "saved"))
            self.rand_model.load()

            self.rand_dir = os.path.join(self.dir, "Rand")

            self.rand_model_dir = os.path.join(self.rand_dir, "models")


            # model variables for plotting
            self.n_eps_rand = int(self.rand_model.model.num_timesteps / self.rand_model.episode_length)
            self.update_freq_rand = self.rand_model.update_freq
            self.n_rand_models = int(self.rand_model.model.num_timesteps / (self.rand_model.episode_length * N_mu * self.update_freq_rand))
            self.rand_episode_checkpoint_multiplier = int(self.rand_model.n_steps / self.rand_model.episode_length)
            # print(f"n_rand_models: {self.n_rand_models}")
        self.seed = self.vanilla_model.seed

        self.train_env = deepcopy(self.vanilla_model.train_env)
        self.train_env_config = deepcopy(vanilla_model_config["env_config"])
        self.train_env_class = deepcopy(self.vanilla_model.env_class)

        self.shifted_envs_range = {}

        self.datasets_range = {}
    
    def add_shifted_env_range(
        self,
        shift_types = ["modeling"],
        shift: dict = {},
        name: str = None
    ):
        if name == None:
            name = "shifted_env_range_" + str(len(self.shifted_envs_range) + 1)
        
        shifted_env_config = deepcopy(self.train_env_config)

        n_models = 1

        logspace = False

        if "modeling" in shift_types:

            n_values = shift["n_values"]
            n_models = shift["n_models"]
            param = shift["param"]

            low = shift["low"]
            high = shift["high"]

            if "logspace" in shift and shift["logspace"]:
                values = np.logspace(np.log10(low), np.log10(high), n_values)
                logspace = True
            else:
                values = np.linspace(low, high, n_values)

            self.shifted_envs_range[name] = (param, values)
            
        self._add_datasets_range(name, n_models)

        self._plot_for_range(name, logspace = logspace)

        # self.save()

    def _add_datasets_range(
        self,
        name,
        n_models
    ):

        shifted_env_config = deepcopy(self.train_env_config)

        shifted_env_config["additional"] = {}

        param = self.shifted_envs_range[name][0]
        values = self.shifted_envs_range[name][1]

        # if self.is_vanilla:
        #     vanilla_dataset = np.zeros(shape = (values.shape[0], n_models), dtype = np.float32)
        # if self.is_rarl:
        #     rarl_dataset = np.zeros(shape = (values.shape[0], n_models), dtype = np.float32)
        # if self.is_rand:
        #     rand_dataset = np.zeros(shape = (values.shape[0], n_models), dtype = np.float32)

        n_seeds = len(self.seeds)

        if self.is_vanilla:
            vanilla_dataset = np.zeros(shape = (values.shape[0], n_seeds), dtype = np.float32)
        if self.is_rarl:
            rarl_dataset = np.zeros(shape = (values.shape[0], n_seeds), dtype = np.float32)
        if self.is_rand:
            rand_dataset = np.zeros(shape = (values.shape[0], n_seeds), dtype = np.float32)
        if self.is_rarl2:
            rarl2_dataset = np.zeros(shape = (values.shape[0], n_seeds), dtype = np.float32)

        for value_iter in range(values.shape[0]):
            
            value = values[value_iter]

            shifted_env_config["additional"][param] = value
            shifted_env_config["initial_infected"] = 0.3

            for seed_iter in range(len(self.seeds)):

                seed = self.seeds[seed_iter]

                if self.is_vanilla:
                    shifted_env = self.train_env_class(**shifted_env_config)
                    vanilla_tester = ModelTester(
                        algo = self.vanilla_model.algorithm, 
                        dir = self.vanilla_dir,
                        test_env = shifted_env,
                        seed = seed
                    )

                    vanilla_tester.load(
                        checkpoint_no = self.n_vanilla_models
                    )
                    
                    vanilla_dataset[value_iter][seed_iter] = vanilla_tester.compute_reward()


                if self.is_rarl:
                    
                    shifted_env = self.train_env_class(**shifted_env_config)
                    rarl_tester = ModelTester(
                        algo = self.rarl_model.gov_algorithm,
                        dir = self.rarl_dir,
                        test_env = shifted_env,
                        seed = seed
                    )

                    rarl_tester.load(
                        checkpoint_no = self.n_rarl_models
                    )
                    
                    rarl_dataset[value_iter][seed_iter] = rarl_tester.compute_reward()

                if self.is_rarl2:
                    
                    shifted_env = self.train_env_class(**shifted_env_config)
                    rarl2_tester = ModelTester(
                        algo = self.rarl2_model.gov_algorithm,
                        dir = self.rarl2_dir,
                        test_env = shifted_env,
                        seed = seed
                    )

                    rarl2_tester.load(
                        checkpoint_no = self.n_rarl2_models
                    )
                    
                    rarl2_dataset[value_iter][seed_iter] = rarl2_tester.compute_reward()

                if self.is_rand:
                    
                    shifted_env = self.train_env_class(**shifted_env_config)
                    rand_tester = ModelTester(
                        algo = self.rand_model.algorithm, 
                        dir = self.rand_dir,
                        test_env = shifted_env,
                        seed = seed
                    )

                    rand_tester.load(
                        checkpoint_no = self.n_rand_models
                    )
                    
                    rand_dataset[value_iter][seed_iter] = rand_tester.compute_reward()

            

        # for value_iter in range(values.shape[0]):

        #     value = values[value_iter]

        #     shifted_env_config["additional"][param] = value
        #     shifted_env_config["initial_infected"] = 0.3

        #     # shifted_env = self.train_env_class(**shifted_env_config)

        #     if self.is_vanilla:
                
        #         shifted_env = self.train_env_class(**shifted_env_config)
        #         vanilla_tester = ModelTester(
        #             algo = self.vanilla_model.algorithm, 
        #             dir = self.vanilla_dir,
        #             test_env = shifted_env,
        #             seed = self.seed
        #         )

        #     if self.is_rarl:
                
        #         shifted_env = self.train_env_class(**shifted_env_config)
        #         rarl_tester = ModelTester(
        #             algo = self.rarl_model.gov_algorithm,
        #             dir = self.rarl_dir,
        #             test_env = shifted_env,
        #             seed = self.seed
        #         )

        #     if self.is_rand:
                
        #         shifted_env = self.train_env_class(**shifted_env_config)
        #         rand_tester = ModelTester(
        #             algo = self.rand_model.algorithm, 
        #             dir = self.rand_dir,
        #             test_env = shifted_env,
        #             seed = self.seed
        #         )

        #     if self.is_rarl2:
                
        #         shifted_env = self.train_env_class(**shifted_env_config)
        #         rarl2_tester = ModelTester(
        #             algo = self.rarl2_model.gov_algorithm,
        #             dir = self.rarl2_dir,
        #             test_env = shifted_env,
        #             seed = self.seed
        #         )
            
        #     for iter in range(n_models):

        #         if self.is_vanilla:

        #             vanilla_tester.load(
        #                 checkpoint_no = int(iter + self.n_vanilla_models - n_models + 1)
        #             )
                    
        #             vanilla_dataset[value_iter][iter] = vanilla_tester.compute_reward()
                
        #         if self.is_rarl:

        #             rarl_tester.load(
        #                 checkpoint_no = int(iter + self.n_rarl_models - n_models + 1)
        #             )
                    
        #             rarl_dataset[value_iter][iter] = rarl_tester.compute_reward()

        #         if self.is_rand:

        #             rand_tester.load(
        #                 checkpoint_no = int(iter + self.n_rand_models - n_models + 1)                    
        #             )
                    
        #             rand_dataset[value_iter][iter] = rand_tester.compute_reward()
                
        #         if self.is_rarl2:

        #             rarl2_tester.load(
        #                 checkpoint_no = int(iter + self.n_rarl2_models - n_models + 1)
        #             )
                    
        #             rarl2_dataset[value_iter][iter] = rarl2_tester.compute_reward()

        if self.is_vanilla:
            self.datasets_range[f"vanilla_{name}"] = vanilla_dataset

        if self.is_rarl:
            self.datasets_range[f"rarl_{name}"] = rarl_dataset

        if self.is_rand:
            self.datasets_range[f"rand_{name}"] = rand_dataset
        
        if self.is_rarl2:
            self.datasets_range[f"rarl2_{name}"] = rarl2_dataset

        if self.is_vanilla:
            with open(
                file = os.path.join(
                    self.vanilla_dir, 
                    f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{values.shape[0]}.json"
                ), 
                mode = "w"
            ) as file:
                json.dump(vanilla_dataset.tolist(), file)

        if self.is_rarl:
            with open(
                file = os.path.join(
                    self.rarl_dir,
                    f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{values.shape[0]}.json"
                ),
                mode = "w"
            ) as file:
                json.dump(rarl_dataset.tolist(), file)

        if self.is_rarl2:
            with open(
                file = os.path.join(
                    self.rarl2_dir,
                    f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{values.shape[0]}.json"
                ),
                mode = "w"
            ) as file:
                json.dump(rarl2_dataset.tolist(), file)

        if self.is_rand:
            with open(
                file = os.path.join(
                    self.rand_dir,
                    f"data/reward_range_on_{name}_n_models_{n_models}_n_values_{values.shape[0]}.json"
                ),
                mode = "w"
            ) as file:
                json.dump(rand_dataset.tolist(), file)

        

    
    def _plot_for_range(
        self,
        name,
        logspace: bool = False
    ):
        values = self.shifted_envs_range[name][1]
        param = self.shifted_envs_range[name][0]

        if self.is_vanilla:
        
            vanilla_dataset = self.datasets_range[f"vanilla_{name}"]
        
            vanilla_means = np.mean(vanilla_dataset, axis = 1)
            vanilla_std_devs = np.std(vanilla_dataset, axis = 1)

        if self.is_rarl:
        
            rarl_dataset = self.datasets_range[f"rarl_{name}"]
        
            rarl_means = np.mean(rarl_dataset, axis = 1)
            rarl_std_devs = np.std(rarl_dataset, axis = 1)

        if self.is_rarl2:
        
            rarl2_dataset = self.datasets_range[f"rarl2_{name}"]
        
            rarl2_means = np.mean(rarl2_dataset, axis = 1)
            rarl2_std_devs = np.std(rarl2_dataset, axis = 1)

        if self.is_rand:
        
            rand_dataset = self.datasets_range[f"rand_{name}"]
        
            rand_means = np.mean(rand_dataset, axis = 1)
            rand_std_devs = np.std(rand_dataset, axis = 1)

        


        plt.ioff()

        if self.is_vanilla:
            plt.errorbar(values, vanilla_means, yerr = vanilla_std_devs, fmt = "o", color = "green", ecolor = "green", capsize = 5)
            plt.plot(values, vanilla_means, color = "green", label = "Vanilla", alpha = 0.6)

        if self.is_rarl:
            plt.errorbar(values, rarl_means, yerr = rarl_std_devs, fmt = "x", color = "blue", ecolor = "blue", capsize = 5)
            plt.plot(values, rarl_means, color = "blue", label = "RARL", alpha = 0.6)

        if self.is_rarl2:
            plt.errorbar(values, rarl2_means, yerr = rarl2_std_devs, fmt = "D", color = "yellow", ecolor = "yellow", capsize = 5)
            plt.plot(values, rarl2_means, color = "yellow", label = "ARL", alpha = 0.6)

        if self.is_rand:
            plt.errorbar(values, rand_means, yerr = rand_std_devs, fmt = "+", color = "brown", ecolor = "brown", capsize = 5)
            plt.plot(values, rand_means, color = "brown", label = "Randomized", alpha = 0.6)

        log_param = ""
        if param in ["alpha", "beta", "gamma"]:
            log_param = "log_"

        if logspace:
            plt.xscale("log")

        plt.xlabel(param)
        plt.ylabel("Rewards")

        plt.title(f"Rewards on {log_param}{param} from {values[0]} to {values[-1]}")

        plt.legend()

        plt.savefig(os.path.join(self.plot_dir, f"{name}_{values[0]:.4f}_to_{values[-1]:.4f}.png"))
        
        plt.close()

    

    # def save(self):

    #     save_dir = os.path.join(self.dir, "saved")
    #     os.makedirs(save_dir, exist_ok = True)

    #     with open(os.path.join(save_dir, "evaluator_range.pkl"), "wb") as file:
    #         pickle.dump(self, file)
        
    #     file_path = os.path.join(save_dir, "evaluator_range.pkl")

    #     print(f"RobustnessEvaluation object saved to {file_path}")
        

    # @staticmethod
    # def load(file_path):
    #     with open(file_path, "rb") as file:
    #         obj = pickle.load(file)
        
    #     print(f"RobustnessEvaluation object loaded from {file_path}")
        
    #     return obj

            