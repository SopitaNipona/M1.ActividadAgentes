import agentpy as ap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple


DIRECTIONS_TO_DELTAS_DICT: dict = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "up-left": (-1, -1),
    "up-right": (-1, 1),
    "down-left": (1, -1),
    "down-right": (1, 1),
}


class BasicVacuumAgent(ap.Agent):
    def setup(self) -> None:
        self.pos: Tuple[int, int] = (1, 1)
        self.i: list = ["idle", (0, 0)]

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> None:
        if percept:
            self.i[0] = "clean"
        else:
            position_delta: Tuple[int, int] = self.model.random.choice(
                list(DIRECTIONS_TO_DELTAS_DICT.values())
            )
            tentative_new_pos: Tuple[int, int] = tuple(
                map(sum, zip(self.pos, position_delta))
            )
            if self.model.is_in_bounds(tentative_new_pos):
                self.i: list = ["move", tentative_new_pos]
            else:
                self.i[0] = "idle"

    def action(self) -> None:
        if self.i[0] == "clean":
            self.model.grid[self.pos[0], self.pos[1]] = 0
            self.record("local_state_utility", 1)
            self.record("movements", 0)
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("local_state_utility", 0)
            self.record("movements", 1)
        elif self.i[0] == "idle":
            self.record("local_state_utility", 0)
            self.record("movements", 0)

    def work(self) -> None:
        self.next(self.see())
        self.action()


class FastVacuumAgent(ap.Agent):
    def setup(self) -> None:
        self.pos: Tuple[int, int] = (1, 1)
        self.i: list = ["idle", (0, 0)]

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> None:
        if percept:
            self.i[0] = "clean"
        else:
            position_delta: Tuple[int, int] = tuple(
                map(
                    sum,
                    zip(
                        *self.model.random.sample(
                            list(DIRECTIONS_TO_DELTAS_DICT.values()), 2
                        )
                    ),
                )
            )
            tentative_new_pos: Tuple[int, int] = tuple(
                map(sum, zip(self.pos, position_delta))
            )
            if self.model.is_in_bounds(tentative_new_pos):
                self.i: list = ["move", tentative_new_pos]
            else:
                self.i[0] = "idle"

    def action(self) -> None:
        if self.i[0] == "clean":
            self.model.grid[self.pos[0], self.pos[1]] = 0
            self.record("local_state_utility", 1)
            self.record("movements", 0)
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("local_state_utility", 0)
            self.record("movements", 1)
        elif self.i[0] == "idle":
            self.record("local_state_utility", 0)
            self.record("movements", 0)

    def work(self) -> None:
        self.next(self.see())
        self.action()


class ExponentialVacuumAgent(ap.Agent):
    def setup(self) -> None:
        self.pos: Tuple[int, int] = (1, 1)
        self.i: list = ["idle", (0, 0), 0]

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> None:
        if percept:
            self.i[0] = "clean"
        else:
            position_delta: Tuple[int, int] = self.model.random.choice(
                list(DIRECTIONS_TO_DELTAS_DICT.values())
            )
            position_delta = tuple(i * 2 ** self.i[2] for i in position_delta)
            self.i[2] += 1
            tentative_new_pos: Tuple[int, int] = tuple(
                map(sum, zip(self.pos, position_delta))
            )
            if self.model.is_in_bounds(tentative_new_pos):
                self.i[0] = "move"
                self.i[1] = tentative_new_pos
            else:
                self.i[0] = "idle"
                self.i[2] = 0

    def action(self) -> None:
        if self.i[0] == "clean":
            self.model.grid[self.pos[0], self.pos[1]] = 0
            self.record("local_state_utility", 1)
            self.record("movements", 0)
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("local_state_utility", 0)
            self.record("movements", 1)
        elif self.i[0] == "idle":
            self.record("local_state_utility", 0)
            self.record("movements", 0)

    def work(self) -> None:
        self.next(self.see())
        self.action()


class LevyVacuumAgent(ap.Agent):
    def setup(self) -> None:
        self.pos: Tuple[int, int] = (1, 1)
        self.i: list = ["idle", (0, 0)]

    def levy_step(self, scale=1.0) -> Tuple[int, int]:
        angle = self.model.random.uniform(0, 2 * np.pi)
        distance = self.model.random.paretovariate(1.5) * scale
        delta_x = int(distance * np.cos(angle))
        delta_y = int(distance * np.sin(angle))
        return delta_x, delta_y

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> None:
        if percept:
            self.i[0] = "clean"
        else:
            position_delta: Tuple[int, int] = self.levy_step()
            tentative_new_pos: Tuple[int, int] = tuple(
                map(sum, zip(self.pos, position_delta))
            )
            if self.model.is_in_bounds(tentative_new_pos):
                self.i = ["move", tentative_new_pos]
            else:
                self.i[0] = "idle"

    def action(self) -> None:
        if self.i[0] == "clean":
            self.model.grid[self.pos[0], self.pos[1]] = 0
            self.record("local_state_utility", 1)
            self.record("movements", 0)
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("local_state_utility", 0)
            self.record("movements", 1)
        elif self.i[0] == "idle":
            self.record("local_state_utility", 0)
            self.record("movements", 0)

    def work(self) -> None:
        self.next(self.see())
        self.action()


class LinearVacuumAgent(ap.Agent):
    def setup(self) -> None:
        self.pos: Tuple[int, int] = (1, 1)
        self.i: list = ["idle", (0, 0), 0]

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> None:
        if percept:
            self.i[0] = "clean"
        else:
            position_delta: Tuple[int, int] = self.model.random.choice(
                list(DIRECTIONS_TO_DELTAS_DICT.values())
            )
            position_delta = tuple(i * self.i[2] for i in position_delta)
            self.i[2] += 1
            tentative_new_pos: Tuple[int, int] = tuple(
                map(sum, zip(self.pos, position_delta))
            )
            if self.model.is_in_bounds(tentative_new_pos):
                self.i[0] = "move"
                self.i[1] = tentative_new_pos
            else:
                self.i[0] = "idle"
                self.i[2] = 0

    def action(self) -> None:
        if self.i[0] == "clean":
            self.model.grid[self.pos[0], self.pos[1]] = 0
            self.record("local_state_utility", 1)
            self.record("movements", 0)
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("local_state_utility", 0)
            self.record("movements", 1)
        elif self.i[0] == "idle":
            self.record("local_state_utility", 0)
            self.record("movements", 0)

    def work(self) -> None:
        self.next(self.see())
        self.action()


class VacuumModel(ap.Model):
    def setup(self) -> None:
        self.grid_size: Tuple[int, int] = (self.p["n"], self.p["m"])
        self.grid: np.ndarray = np.zeros(self.grid_size, dtype=int)
        self.grid[:] = np.random.choice(
            [0, 1],
            size=self.grid_size,
            p=[1 - self.p["dirt_percentage"], self.p["dirt_percentage"]],
        )
        self.agents: ap.AgentList = ap.AgentList(self, self.p["k0"], BasicVacuumAgent)
        self.agents.extend(ap.AgentList(self, self.p["k1"], FastVacuumAgent))
        self.agents.extend(ap.AgentList(self, self.p["k2"], ExponentialVacuumAgent))
        self.agents.extend(ap.AgentList(self, self.p["k3"], LevyVacuumAgent))
        self.agents.extend(ap.AgentList(self, self.p["k4"], LinearVacuumAgent))

    def step(self) -> None:
        self.agents.work()
        if self.grid.sum() == 0:
            self.stop()

    def end(self):
        self.report("time_used", self.t)
        self.report("clean_percentage", self.clean_percentage())
        movements_made: int = sum(
            sum(agent.log["movements"])
            for agent in self.agents
            if "movements" in agent.log
        )
        self.report("movements_made", movements_made)
        self.report(
            "k",
            sum((self.p["k0"], self.p["k1"], self.p["k2"], self.p["k3"], self.p["k4"])),
        )
        for agent in self.agents:
            self.report(
                f"agent_{agent.id}_utility_log",
                agent.log.get("local_state_utility", []),
            )
        average_of_long_term_utilities: float = sum(
            sum(agent.log["local_state_utility"])
            for agent in self.agents
            if "local_state_utility" in agent.log
        ) / len(self.agents)
        self.report("average_long_term_utility", average_of_long_term_utilities)

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        ans: bool = True
        for i in range(len(pos)):
            ans = ans and 0 <= pos[i] < self.grid_size[i]
        return ans

    def clean_percentage(self) -> float:
        return (1 - np.sum(self.grid) / self.grid.size) * 100


# Experiment 1

probability_matrix: np.ndarray = np.zeros((5, 4))
average_utility_matrix: np.ndarray = np.zeros((5, 4))
number_of_experiments0: int = 50  # 50
for i in range(5):
    for j in range(4):

        params0: dict = {
            "n": 100,
            "m": 100,
            "k0": 0,
            "k1": 0,
            "k2": 0,
            "k3": 0,
            "k4": 0,
            "dirt_percentage": 0.25,
            "steps": 30000 * (j + 1) // 4,  # percentage of t_max
        }
        params0[f"k{i}"] = 20
        sample0: ap.Sample = ap.Sample(params0)
        experiment0: ap.Experiment = ap.Experiment(
            VacuumModel, sample0, iterations=number_of_experiments0, record=True
        )
        results0: ap.datadict.DataDict = experiment0.run()
        results_df0: pd.DataFrame = pd.DataFrame(results0.reporters)
        probability_matrix[i, j] = (
            results_df0["clean_percentage"].eq(100).sum() / number_of_experiments0
        )
        average_utility_matrix[i, j] = results_df0["average_long_term_utility"].mean()

print(probability_matrix)
sns.heatmap(probability_matrix, annot=True, fmt=".2f")
plt.title(
    "Probability of Success (100% Cleaned) for Different Kinds of Agents and Runs"
)
plt.xlabel("Run")
plt.ylabel("Agent")
plt.show()


# TODO: Deben calcular el agente óptimo, o en su caso, el agente con mayor probabilidad de éxito.

print(average_utility_matrix)
sns.heatmap(average_utility_matrix, annot=True, fmt=".2f")
plt.title("Average Long Term Utility for Different Kinds of Agents and Runs")
plt.xlabel("Run")
plt.ylabel("Agent")
plt.show()

# Experiment 2

number_of_samples = 50
max_agents = 100
all_results = []
for i in range(5):
    params = {
        "n": 100,
        "m": 100,
        "k0": 0,
        "k1": 0,
        "k2": 0,
        "k3": 0,
        "k4": 0,
        "dirt_percentage": 0.25,
        "steps": 100000,  # t_max
    }
    params[f"k{i}"] = ap.IntRange(1, max_agents)
    sample = ap.Sample(params, n=number_of_samples)
    experiment = ap.Experiment(VacuumModel, sample, iterations=1, record=True)
    results = experiment.run()
    results_df = pd.DataFrame(results.reporters)
    results_df["Agent_Type"] = f"k{i}"
    all_results.append(results_df)

final_results_df = pd.concat(all_results, ignore_index=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=final_results_df, x="k", y="movements_made", hue="Agent_Type", palette="Set1"
)
plt.title("Number of Agents vs. Movements Made")
plt.xlabel("Number of Agents")
plt.ylabel("Movements Made")
plt.legend(title="Agent Type")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=final_results_df, x="k", y="time_used", hue="Agent_Type", palette="Set1"
)
plt.title("Number of Agents vs. Time Used")
plt.xlabel("Number of Agents")
plt.ylabel("Time Used")
plt.legend(title="Agent Type")
plt.show()

# Experiment 3

number_of_agents2: int = 20
for i in range(5):
    params2: dict = {
        "n": 100,
        "m": 100,
        "k0": 0,
        "k1": 0,
        "k2": 0,
        "k3": 0,
        "k4": 0,
        "dirt_percentage": 0.25,
        "steps": 100000,  # t_max
    }
    params2[f"k{i}"] = number_of_agents2
    sample2: ap.Sample = ap.Sample(params2)
    experiment2: ap.Experiment = ap.Experiment(
        VacuumModel, sample2, iterations=1, record=True
    )
    results2: ap.datadict.DataDict = experiment2.run()
    results_df2: pd.DataFrame = pd.DataFrame(results2.reporters)

    cumulative_utility = []
    for agent_id in range(number_of_agents2):
        key = f"agent_{agent_id}_utility_log"
        if key in results_df2.columns:
            utility_list = results_df2[key].iloc[0]
            cumulative_utility.append(np.cumsum(utility_list))

    plt.figure()
    for idx, utility in enumerate(cumulative_utility):
        plt.plot(utility, label=f"Agent {idx}")

    plt.title(f"Cumulative Utility for Agent Type {i}")
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Utility")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
