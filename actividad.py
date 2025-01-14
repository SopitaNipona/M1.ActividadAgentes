import agentpy as ap 
import numpy as np
from typing import Tuple

# Define base CleaningAgent
class CleaningAgent(ap.Agent):
    def setup(self) -> None:
        self.moves = 0
        self.cleaned_cells = 0  # Track how many cells the agent cleans

    def see(self) -> int:
        # Check if the current cell is dirty
        return self.model.grid[self.pos[0], self.pos[1]]

    def next(self, percept: int) -> int:
        # Return the percept as internal state
        return percept

    def action(self, internal_state: int) -> None:
        if internal_state:  # If the cell is dirty
            self.model.grid[self.pos[0], self.pos[1]] = 0  # Clean the cell
            self.cleaned_cells += 1
        else:
            # Move to a random adjacent position
            move = self.model.random.choice(
                [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
            )
            new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
            if self.model.is_valid_position(new_pos):
                self.pos = new_pos
                self.moves += 1

# Define specialized agents
class FastCleaningAgent(CleaningAgent):
    def action(self, internal_state: int) -> None:
        if internal_state:  # If the cell is dirty
            self.model.grid[self.pos[0], self.pos[1]] = 0  # Clean the cell
            self.cleaned_cells += 1
        else:
            # Move to two random adjacent positions (faster movement)
            for _ in range(2):
                move = self.model.random.choice(
                    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
                )
                new_pos = (self.pos[0] + move[0], self.pos[1] + move[1])
                if self.model.is_valid_position(new_pos):
                    self.pos = new_pos
                    self.moves += 1

class StrategicCleaningAgent(CleaningAgent):
    def action(self, internal_state: int) -> None:
        if internal_state:  # If the cell is dirty
            self.model.grid[self.pos[0], self.pos[1]] = 0  # Clean the cell
            self.cleaned_cells += 1
        else:
            # Prefer moving toward dirtier areas
            dirt_positions = np.argwhere(self.model.grid == 1)
            if len(dirt_positions) > 0:
                target = self.model.random.choice(dirt_positions)
                new_pos = (target[0], target[1])
                if self.model.is_valid_position(new_pos):
                    self.pos = new_pos
                    self.moves += 1

class CleaningModel(ap.Model):
    def setup(self) -> None:
        self.grid_size: Tuple[int, int] = self.p["grid_size"]
        self.grid = np.zeros(self.grid_size, dtype=int)
        self.grid[:] = np.random.choice(
            [0, 1],
            size=self.grid_size,
            p=[1 - self.p["dirt_percentage"], self.p["dirt_percentage"]],
        )
        self.initial_grid_state = self.grid.tolist()  # Save initial grid as a list

        # Create agents with different types
        self.agents = ap.AgentList(self, self.p["n_agents"] // 3, CleaningAgent)
        self.fast_agents = ap.AgentList(self, self.p["n_agents"] // 3, FastCleaningAgent)
        self.strategic_agents = ap.AgentList(self, self.p["n_agents"] - 2 * (self.p["n_agents"] // 3), StrategicCleaningAgent)
        self.agents += self.fast_agents + self.strategic_agents  # Combine all agent types

        for agent in self.agents:
            agent.pos = (self.random.randint(0, self.grid_size[0]), self.random.randint(0, self.grid_size[1]))
            while not self.is_valid_position(agent.pos):
                agent.pos = (self.random.randint(0, self.grid_size[0]), self.random.randint(0, self.grid_size[1]))
        self.cleaning_start = self.grid.sum()

    def step(self) -> None:
        for agent in self.agents:
            agent.action(agent.next(agent.see()))
        if self.grid.sum() == 0 or self.t >= self.p["max_steps"]:
            self.stop()

    def end(self):
        total_moves = sum(agent.moves for agent in self.agents)
        final_grid_state = self.grid.tolist()  # Save final grid as a list

        print("Initial grid as list:")
        print(self.initial_grid_state)

        print("Final grid as list:")
        print(final_grid_state)

        print(f"Simulation ended after {self.t} steps.")
        print(f"Cleaning percentage: {self.clean_percentage():.2f}%")
        print(f"Total moves by all agents: {total_moves}")

        # Individual performance metrics
        for i, agent in enumerate(self.agents):
            print(f"Agent {i + 1}: {agent.moves} moves, {agent.cleaned_cells} cells cleaned")

        # Probabilities for each run (A, B, C, D)
        total_cells = self.cleaning_start  # Total dirty cells at the start
        results = {0.25: 0, 0.5: 0, 0.75: 0, 1.0: 0}

        for agent in self.agents:
            cleaned_percentage = agent.cleaned_cells / total_cells
            print(f"Agent cleaned {cleaned_percentage * 100:.2f}% of total cells.")  # Debugging percentage
            for threshold in sorted(results.keys()):
                if cleaned_percentage >= threshold:
                    results[threshold] += 1

        print("Summary of agents meeting cleaning thresholds:")
        for threshold, count in sorted(results.items()):
            print(f"Agents cleaning at least {threshold * 100}%: {count}")

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]

    def clean_percentage(self) -> float:
        return (1 - np.sum(self.grid) / self.cleaning_start) * 100

# Parameters for the simulation
params = {
    "grid_size": (10, 10),
    "n_agents": 6,  # Updated to 6 agents
    "dirt_percentage": 0.2,
    "max_steps": 1000,
}

model = CleaningModel(params)
model.run()

