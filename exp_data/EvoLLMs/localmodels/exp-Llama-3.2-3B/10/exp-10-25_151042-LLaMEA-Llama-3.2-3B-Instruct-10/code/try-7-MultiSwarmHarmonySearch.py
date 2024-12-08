import numpy as np
from scipy.optimize import minimize

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm = self.initialize_swarm()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.line_search = False

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
                if r1 < 0.2:
                    self.swarm[i] = self.swarm[i] + 0.1 * (self.swarm[i] - self.best_solution)
                elif r2 < 0.4:
                    self.swarm[i] = self.swarm[i] - 0.1 * (self.swarm[i] - self.best_solution)
                elif r1 < 0.6:
                    self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                elif r2 < 0.8:
                    self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

            # Harmony search phase
            for i in range(len(self.swarm)):
                if np.random.rand() < 0.01:
                    r1, r2 = random.random(), random.random()
                    if r1 < 0.5:
                        self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                    else:
                        self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

            # Line search phase
            if self.line_search:
                direction = self.swarm[i] - self.best_solution
                step_size = 0.1
                while True:
                    new_solution = self.swarm[i] + step_size * direction
                    new_fitness = func(new_solution)
                    if new_fitness < self.best_fitness:
                        self.swarm[i] = new_solution
                        self.best_solution = new_solution
                        self.best_fitness = new_fitness
                    else:
                        step_size *= 0.9
                        if step_size < 1e-6:
                            break

        return self.best_solution, self.best_fitness

def func(solution):
    # Example noiseless function
    return np.sum(solution**2)

# Example usage
ms = MultiSwarmHarmonySearch(100, 10)
best_solution, best_fitness = ms(func)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")