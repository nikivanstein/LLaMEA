import numpy as np
import random
import copy

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm = self.initialize_swarm()
        self.best_solution = None
        self.best_fitness = float('inf')
        self.adaptation_prob = 0.1
        self.refine_prob = 0.1

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
                if random.random() < self.adaptation_prob:
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
                if random.random() < self.refine_prob:
                    r1, r2 = random.random(), random.random()
                    if r1 < 0.5:
                        self.swarm[i] = self.swarm[i] + 0.1 * np.random.uniform(-1, 1, self.dim)
                    else:
                        self.swarm[i] = self.swarm[i] - 0.1 * np.random.uniform(-1, 1, self.dim)

        # Refine the best solution
        new_solution = copy.deepcopy(self.best_solution)
        for _ in range(int(self.budget * self.refine_prob)):
            r1, r2 = random.random(), random.random()
            if r1 < 0.5:
                new_solution = new_solution + 0.1 * np.random.uniform(-1, 1, self.dim)
            else:
                new_solution = new_solution - 0.1 * np.random.uniform(-1, 1, self.dim)
            if func(new_solution) < func(self.best_solution):
                self.best_solution = new_solution
                self.best_fitness = func(self.best_solution)

        return self.best_solution, self.best_fitness

def func(solution):
    # Example noiseless function
    return np.sum(solution**2)

# Example usage
ms = MultiSwarmHarmonySearch(100, 10)
best_solution, best_fitness = ms(func)
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")