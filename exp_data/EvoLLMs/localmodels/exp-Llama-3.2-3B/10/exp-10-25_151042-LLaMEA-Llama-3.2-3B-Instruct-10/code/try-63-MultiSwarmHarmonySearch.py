import numpy as np
import random

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm = self.initialize_swarm()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.adaptation_probability = 0.1

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.budget):
            solution = np.random.uniform(-5.0, 5.0, self.dim)
            swarm.append(solution)
        return swarm

    def fitness(self, solution):
        return func(solution)

    def __call__(self, func):
        for _ in range(self.budget):
            if self.best_fitness > func(self.swarm[0]):
                self.best_solution = self.swarm[0]
                self.best_fitness = func(self.swarm[0])

            for i, solution in enumerate(self.swarm):
                fitness = func(solution)
                if fitness < self.best_fitness:
                    self.swarm[i] = solution
                    self.best_fitness = fitness

            # Adaptation phase
            for i in range(len(self.swarm)):
                r1, r2 = random.random(), random.random()
                if r1 < self.adaptation_probability:
                    # Random mutation
                    self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                elif r2 < self.adaptation_probability * 2:
                    # Random improvement
                    r3, r4 = random.random(), random.random()
                    if r3 < 0.5:
                        self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                    else:
                        self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

            # Harmony search phase
            for i in range(len(self.swarm)):
                if np.random.rand() < 0.01:
                    # Random exploration
                    r1, r2 = random.random(), random.random()
                    if r1 < 0.5:
                        self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                    else:
                        self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

        return self.best_solution, self.best_fitness

def func(solution):
    # Example noiseless function
    return np.sum(solution**2)

# Example usage
ms = MultiSwarmHarmonySearch(100, 10)
best_solution, best_fitness = ms(func)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")