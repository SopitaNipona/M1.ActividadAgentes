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
        self.long_term_utility: int = 0

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
            self.long_term_utility += 1
        elif self.i[0] == "move":
            self.pos: Tuple[int, int] = self.i[1]
            self.record("movements", 1)
        elif self.i[0] == "idle":
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
        self.agents: ap.AgentList = ap.AgentList(
            self, self.p["k0"], BasicVacuumAgent
        )  # TODO: Add 4 kinds of agents

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

    def is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        ans: bool = True
        for i in range(len(pos)):
            ans = ans and 0 <= pos[i] < self.grid_size[i]
        return ans

    def clean_percentage(self) -> float:
        return (1 - np.sum(self.grid) / self.grid.size) * 100


# TODO: Deben de calcular probabilidades de cada tipo de agente sobre las siguientes corridas:
"""
Corrida A: Corrida de los agentes cuando limpian todas las celdas hasta el 25% del tiempo máximo, no después.
Corrida B: Corrida de los agentes cuando limpian todas las celdas hasta el 50% del tiempo máximo, no después.
Corrida C: Corrida de los agentes cuando limpian todas las celdas hasta el 75% del tiempo máximo, no después.
Corrida D: Corrida de los agentes cuando limpian todas las celdas hasta el 100% del tiempo máximo.
"""

# TODO: Deben calcular el agente óptimo, o en su caso, el agente con mayor probabilidad de éxito.

# DONE: Analiza cómo la cantidad de agentes impacta el tiempo dedicado, así como la cantidad de movimientos realizados. 

params: dict = {
    "n": 100,
    "m": 100,
    "k0": ap.IntRange(1, 100),
    "k1": 0,
    "k2": 0,
    "k3": 0,
    "k4": 0,
    "dirt_percentage": 0.25,
    "steps": 100000,  # t_max
}

sample: ap.Sample = ap.Sample(params, n=20)
experiment: ap.Experiment = ap.Experiment(
    VacuumModel, sample, iterations=1, record=True
)
results: ap.datadict.DataDict = experiment.run()
results_df: pd.DataFrame = pd.DataFrame(results.reporters)

sns.scatterplot(data=results_df, x="k", y="movements_made")
plt.title("Number of Agents vs. Movements Made")
plt.xlabel("Number of Agents")
plt.ylabel("Movements Made")
plt.show()

sns.scatterplot(data=results_df, x="k", y="time_used")
plt.title("Number of Agents vs. Time Used")
plt.xlabel("Number of Agents")
plt.ylabel("Time Used")
plt.show()

# TODO: También analiza el desempeño de los agentes.
