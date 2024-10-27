import numpy as np

class MultiSwarmDE_LS_Optimizer:
    def __init__(self, budget, dim, pop_size=20, f=0.5, cr=0.9, ls_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.ls_prob = ls_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))

        def differential_evolution(population, func):
            new_population = []
            for idx, target in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                donor = a + self.f * (b - c)
                mask = np.random.rand(self.dim) < self.cr
                trial = np.where(mask, donor, target)
                if func(trial) < func(target):
                    new_population.append(trial)
                else:
                    new_population.append(target)
            return np.array(new_population)

        def local_search(population, func):
            new_population = []
            for individual in population:
                if np.random.rand() < self.ls_prob:
                    candidate = individual + np.random.normal(0, 1, size=self.dim)
                    if func(candidate) < func(individual):
                        new_population.append(candidate)
                    else:
                        new_population.append(individual)
                else:
                    new_population.append(individual)
            return np.array(new_population)

        population = initialize_population()
        remaining_budget = self.budget - self.pop_size

        while remaining_budget > 0:
            population = differential_evolution(population, func)
            population = local_search(population, func)
            remaining_budget -= self.pop_size

        best_individual = population[np.argmin([func(p) for p in population])]
        return best_individual

# Example usage:
# optimizer = MultiSwarmDE_LS_Optimizer(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function