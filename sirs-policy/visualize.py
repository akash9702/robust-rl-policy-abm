from sirs_gov import SIRSGov
from model import VanillaModelManager, RARLModelManager, ModelTester
from copy import deepcopy
import os
import json
import numpy as np
import dill as picklefrom 
from copy import deepcopy
import os
import json
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


class VisualizePolicy:

    actions_to_names = {
        "lockdown" : "Lockdown Policy"
    }

    observations_to_names = {
        "n_s" : "Number of Susceptible",
        "n_i" : "Number of Infected"
    }

    def __init__(
        self,
        observation_types: list = ["n_s", "n_i"],
        observation_ranges: dict = {"n_s": (0, 100), "n_i": (0, 100)},
        # action_types: list = ["lockdown"],
        n_values: int = 101,
        test_env = None
    ):
        self.observation_types = observation_types
        self.observation_ranges = observation_ranges
        self.n_values = n_values
        # self.action_types = action_types

        if test_env is None:
            raise ValueError("please provide test_env to VisualizePolicy")
        self.test_env = test_env

        self._generate_grid()

    def visualize_policy(
        self,
        model,
        plot_dir,
        name: str = ""
        # env_path=None
    ):
        os.makedirs(plot_dir, exist_ok=True)
        
        # env = None

        # if env_path is not None:
        #     env = deepcopy(self.test_env)
        #     env = DummyVecEnv([lambda: env])
        #     env = VecNormalize.load(load_path=env_path, venv=env)

        # recovered_values = [0, 20, 40, 60, 80, 100]

        # for r in recovered_values:


        fig = self._create_heatmap(model)
        
        fig.savefig(f"{plot_dir}/policy_visualization.png", dpi=300, bbox_inches='tight')

        if name != "":
            os.makedirs(f"{plot_dir}/../../../visualization_plots", exist_ok = True)
            fig.savefig(f"{plot_dir}/../../../visualization_plots/{name}.png", dpi = 300)
        

    def _generate_grid(self):
        # Generate a grid of points between (0, 0) and (100, 100)
        x_vals = np.linspace(
            self.observation_ranges[self.observation_types[0]][0], 
            self.observation_ranges[self.observation_types[0]][1], 
            self.n_values
        )
        y_vals = np.linspace(
            self.observation_ranges[self.observation_types[1]][0], 
            self.observation_ranges[self.observation_types[1]][1], 
            self.n_values
        )
        self.x_grid, self.y_grid = np.meshgrid(x_vals, y_vals)

    def _create_heatmap(self, model):
        plt.ioff()

        policy_values = np.zeros((self.n_values, self.n_values))
        policy_values -= 1

        for i in range(self.n_values):
            for j in range(self.n_values):
                # obs = np.array([[self.x_grid[i, j], self.y_grid[i, j]]])

                non_recovered = int(self.x_grid[i, j]) + int(self.y_grid[i, j])

                if non_recovered < 100:
                    
                    obs = (int(self.x_grid[i, j]), int(self.y_grid[i, j]), 100 - non_recovered)

                    prediction, _ = model.predict(obs, deterministic=True)
                    policy_values[i, j] = prediction
                
                




                # if int(self.x_grid[i, j]) + int(self.y_grid[i, j]) + r == 100:

                #     obs = (int(self.x_grid[i, j]), int(self.y_grid[i, j]), r)
                #     # print(obs)

                #     # Predict action (0 or 1 for lockdown) based on the model
                #     prediction, _ = model.predict(obs, deterministic=True)
                #     policy_values[i, j] = prediction
        
        plt.rcParams.update({'font.size': 16, 'axes.labelsize': 16})

        fig, ax = plt.subplots(figsize=(4, 4))

        # Replace -1 values with NaN to ensure they appear as whitespace
        policy_values_masked = np.where(policy_values == -1, np.nan, policy_values)

                # Policy value red/blue grid with masked values
        im = ax.imshow(
            policy_values_masked, 
            extent=(
                self.observation_ranges[self.observation_types[0]][0],
                self.observation_ranges[self.observation_types[0]][1], 
                self.observation_ranges[self.observation_types[1]][0], 
                self.observation_ranges[self.observation_types[1]][1]
            ),
            origin='lower',
            aspect='auto',
            vmin=0,
            vmax=1,
            cmap='coolwarm'  # 'coolwarm' gives us red/blue colors
        )

        # # Policy value red/blue grid
        # im = ax.imshow(
        #     policy_values, 
        #     extent=(
        #         self.observation_ranges[self.observation_types[0]][0],
        #         self.observation_ranges[self.observation_types[0]][1], 
        #         self.observation_ranges[self.observation_types[1]][0], 
        #         self.observation_ranges[self.observation_types[1]][1]
        #     ),
        #     origin='lower',
        #     aspect='auto',
        #     # vmin=0,
        #     # vmax=1,
        #     cmap='coolwarm'  # 'coolwarm' gives us red/blue colors
        # )

        # ax.set_title(self.actions_to_names[self.action_types[0]])
        ax.set_xlabel(self.observations_to_names[self.observation_types[0]])
        ax.set_ylabel(self.observations_to_names[self.observation_types[1]])

        plt.tight_layout()

        # Create custom legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Lockdown')
        blue_patch = mpatches.Patch(color='blue', label='No Lockdown')
        plt.legend(handles=[red_patch, blue_patch], loc='upper right', bbox_to_anchor=(1.2, 1.1))

        plt.close()

        return fig


        # # Policy value heatmap
        # im = ax.imshow(
        #     policy_values, 
        #     extent=(
        #         self.observation_ranges[self.observation_types[0]][0],
        #         self.observation_ranges[self.observation_types[0]][1], 
        #         self.observation_ranges[self.observation_types[1]][0], 
        #         self.observation_ranges[self.observation_types[1]][1]
        #     ),
        #     origin='lower',
        #     aspect='auto',
        #     vmin=0,
        #     vmax=1,
        #     cmap='viridis'
        # )

        # ax.set_title(self.actions_to_names[self.action_types[0]])
        # ax.set_xlabel(self.observations_to_names[self.observation_types[0]])
        # ax.set_ylabel(self.observations_to_names[self.observation_types[1]])

        # plt.tight_layout()

        # # Colorbar
        # # fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04, ticks=np.linspace(0, 1, 11))

        # return fig



