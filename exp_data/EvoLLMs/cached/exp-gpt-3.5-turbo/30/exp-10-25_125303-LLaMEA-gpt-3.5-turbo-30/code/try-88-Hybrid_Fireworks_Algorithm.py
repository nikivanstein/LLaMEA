import numpy as np

class Hybrid_Fireworks_Algorithm:
    def __init__(self, budget, dim, population_size=30, f=0.5, cr=0.9, hmcr=0.7, par=0.4, mw=0.2, mutation_prob=0.3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.hmcr = hmcr
        self.par = par
        self.mw = mw
        self.mutation_prob = mutation_prob

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def fireworks_step(population, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                leader_idx = np.argmin([np.linalg.norm(ind - target) for ind in population])
                leader = population[leader_idx]

                for i in range(len(target)):
                    step_size = np.linalg.norm(target - leader)
                    if np.random.rand() < self.hmcr:
                        if np.random.rand() < self.par:
                            target[i] += step_size * np.random.uniform(0, 1)
                        else:
                            target[i] = leader[i]
                    if np.random.rand() < self.mw:
                        target[i] += np.random.uniform(-1, 1)
                    if np.random.rand() < self.mutation_prob:
                        target[i] += np.random.normal(0, 0.1)  # Adaptive mutation
                new_population.append(target)
            return np.array(new_population)

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = fireworks_step(population, best_individual)
            for idx, individual in enumerate(new_population):
                if remaining_budget <= 0:
                    break
                new_fitness = func(individual)
                if new_fitness < func(population[idx]):
                    population[idx] = individual
                    if new_fitness < func(best_individual):
                        best_individual = individual
                remaining_budget -= 1

        return best_individual

# Example usage:
# optimizer = Hybrid_Fireworks_Algorithm(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function