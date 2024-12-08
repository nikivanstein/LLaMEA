import numpy as np

class Adaptive_Evo_Multi_Strategy:
    def __init__(self, budget, dim, population_size=30, f=0.5, cr=0.9, diversity_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.diversity_rate = diversity_rate

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def evo_step(population, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                mutant = target + self.f * (population[np.random.randint(0, self.population_size)] - target)
                crossover_points = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_points, mutant, target)

                for i in range(len(target)):
                    if np.random.rand() < self.diversity_rate:
                        trial[i] += np.random.uniform(-1, 1)  # Enhancing diversity
                new_population.append(trial)
            return np.array(new_population)

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = evo_step(population, best_individual)
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
# optimizer = Adaptive_Evo_Multi_Strategy(budget=1000, dim=10)
# result = optimizer(lambda x: np.sum(x**2))  # Optimize the sphere function