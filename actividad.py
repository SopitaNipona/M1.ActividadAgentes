import agentpy as ap
import numpy as np
from typing import Tuple


class CleaningAgent(ap.Agent):
    def setup(self) -> None:
        self.internal_state: int = 0

    def see(self) -> int:
        # Check if the current cell is dirty
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> int:
        # Update internal state
        self.internal_state = percept
        return self.internal_state

    def action(self, internal_state: int) -> None:
        if internal_state:  # If the cell is dirty
            self.model.grid[self.pos[0], self.pos[1]] = 0  # Clean the cell
        else:
            # Move to a random adjacent position
            move = self.model.random.choice(
                [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            )
            new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
            if self.model.is_valid_position(new_pos):
                self.pos = new_pos
                self.moves += 1
        print("Agent position: ", self.pos)


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
        self.agents.moves = 0
        print(self.grid)

    def step(self) -> None:
        for agent in self.agents:
            agent.action(agent.next(agent.see()))
        if self.grid.sum() == 0:
            self.stop()
            
    def end(self):
        print(f"Simulation ended after {self.t} steps.")
        print(f"Cleaning percentage: {self.clean_percentage():.2f}%")
        print(self.grid)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def clean_percentage(self) -> float:
        return (1 - np.sum(self.grid) / self.grid.size) * 100


# Parameters for the simulation
params = {
    "grid_size": (10, 10),
    "n_agents": 1,
    "dirt_percentage": 1,
    "steps": 10000,
}

model = CleaningModel(params)
model.run()
