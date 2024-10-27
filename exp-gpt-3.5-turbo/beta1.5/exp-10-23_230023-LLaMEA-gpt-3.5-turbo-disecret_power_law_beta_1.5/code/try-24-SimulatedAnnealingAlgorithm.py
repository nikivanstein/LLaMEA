import numpy as np

class SimulatedAnnealingAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.init_temperature = 100.0
        self.cooling_rate = 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def accept(self, current_fitness, new_fitness, temperature):
        if new_fitness < current_fitness:
            return True
        return np.random.rand() < np.exp((current_fitness - new_fitness) / temperature)

    def perturb(self, x):
        return x + np.random.uniform(-0.1, 0.1, size=self.dim)

    def __call__(self, func):
        current_solution = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        current_fitness = func(current_solution)
        temperature = self.init_temperature

        for _ in range(self.budget):
            new_solution = self.perturb(current_solution)
            new_fitness = func(new_solution)

            if self.accept(current_fitness, new_fitness, temperature):
                current_solution = new_solution
                current_fitness = new_fitness

            temperature *= self.cooling_rate

        return current_solution