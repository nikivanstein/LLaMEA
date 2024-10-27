import numpy as np

class Novel_Metaheuristic_Algorithm:
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

        def novel_step(population, best_individual):
            # Insert novel algorithmic steps here
            pass

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = novel_step(population, best_individual)
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