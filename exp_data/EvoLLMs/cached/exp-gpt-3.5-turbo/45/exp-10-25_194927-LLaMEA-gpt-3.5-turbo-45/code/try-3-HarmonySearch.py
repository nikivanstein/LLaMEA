import numpy as np

class HarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.3
        self.bandwidth = 0.01

    def __call__(self, func):
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.budget):
            new_solution = np.zeros(self.dim)
            for d in range(self.dim):
                if np.random.rand() < self.hmcr:
                    new_solution[d] = harmony_memory[np.random.randint(self.budget), d]
                    if np.random.rand() < self.par:
                        new_solution[d] += self.bandwidth * np.random.normal()
                else:
                    new_solution[d] = np.random.uniform(self.lower_bound, self.upper_bound)
            
            fitness = func(new_solution)
            if fitness < best_fitness:
                best_solution = new_solution
                best_fitness = fitness
            
            harmony_memory[np.argmax(fitness)] = new_solution

        return best_solution