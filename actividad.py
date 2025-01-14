import agentpy as ap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple

class CleaningAgent(ap.Agent):
    def setup(self) -> None:
        self.internal_state: int = 0
        self.moves = 0

    def see(self) -> int:
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> int:
        self.internal_state = percept
        return self.internal_state

    def action(self, internal_state: int) -> None:
        if internal_state:
            self.model.grid[self.pos[0], self.pos[1]] = 0
        else:
            move = self.model.random.choice(
                [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            )
            new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
            if self.model.is_valid_position(new_pos):
                self.pos = new_pos
                self.moves += 1
                self.record('moves', self.moves)


class CleaningModel(ap.Model):
    def setup(self) -> None:
        self.grid_size: Tuple[int, int] = self.p["grid_size"]
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.grid[:] = np.random.choice(
            [0, 1],
            size=self.grid_size,
            p=[1 - self.p["dirt_percentage"], self.p["dirt_percentage"]],
        )
        self.agents = ap.AgentList(self, self.p["n_agents"], CleaningAgent)
        self.agents.pos = (1, 1)

    def step(self) -> None:
        for agent in self.agents:
            agent.action(agent.next(agent.see()))
        if self.grid.sum() == 0:
            self.stop()
            
    def end(self):
        total_steps = sum(agent.moves for agent in self.agents)
        self.report('total_steps', total_steps)
        self.report('n_agents', self.p["n_agents"])
        self.report('steps_used', self.t)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def clean_percentage(self) -> float:
        return (1 - np.sum(self.grid) / self.grid.size) * 100

# Parameters for the simulation
params = {
    "grid_size": (100, 100),
    "n_agents": ap.IntRange(1, 1000),
    "dirt_percentage": 1,
    "steps": 100000,
}

sample = ap.Sample(params, n=10)
experiment = ap.Experiment(CleaningModel, sample, iterations=1, record=True)
results = experiment.run()

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results.reporters)

# Crear gráfica de número de agentes vs. pasos totales
sns.scatterplot(data=results_df, x='n_agents', y='total_steps')
plt.title('Número de Agentes vs. Pasos Totales')
plt.xlabel('Número de Agentes')
plt.ylabel('Pasos Totales')
plt.show()

# Crear gráfica de número de agentes vs. tiempo usado con escala logarítmica
sns.scatterplot(data=results_df, x='n_agents', y='steps_used')
plt.yscale('log')
plt.title('Número de Agentes vs. Tiempo Usado (Escala Logarítmica)')
plt.xlabel('Número de Agentes')
plt.ylabel('Tiempo Usado (Pasos)')
plt.show()
