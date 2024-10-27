import numpy as np

class PSO_DE_Hybrid:
    def __init__(self, budget, dim, population_size=30, f=0.5, cr=0.9, w=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.w = w

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def pso_de_step(population):
            new_population = []
            for idx, target in enumerate(population):
                pbest = population[np.argmin([func(ind) for ind in population])]
                gbest = population[np.argmin([func(ind) for ind in population])]
                v = np.random.uniform(0, 1, size=self.dim) * v + self.w * (pbest - target) + self.cr * (gbest - target)
                trial = target + self.f * v
                new_population.append(trial)
            return np.array(new_population)

        population = initialize_population()
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = pso_de_step(population)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                remaining_budget -= 1

        return population[np.argmin([func(ind) for ind in population]])

# Example usage:
# optimizer = PSO_DE_Hybrid(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function