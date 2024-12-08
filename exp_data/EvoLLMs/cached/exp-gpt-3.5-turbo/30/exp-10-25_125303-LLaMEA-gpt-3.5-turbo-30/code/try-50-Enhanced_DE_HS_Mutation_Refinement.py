import numpy as np

class Enhanced_DE_HS_Mutation_Refinement:
    def __init__(self, budget, dim, population_size=30, f=0.5, cr=0.9, hmcr=0.7, par=0.4, mw=0.2):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.hmcr = hmcr
        self.par = par
        self.mw = mw

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        def de_hs_step(population, best_individual):
            new_population = []
            for idx, target in enumerate(population):
                mutant = target + self.f * (population[np.random.randint(0, self.population_size)] - target)
                crossover_points = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_points, mutant, target)

                for i in range(len(target)):
                    if np.random.rand() < self.hmcr:
                        if np.random.rand() < self.par:
                            trial[i] = best_individual[i]
                        else:
                            idx = np.random.choice(self.population_size)
                            trial[i] = population[idx][i]
                    if np.random.rand() < self.mw:
                        trial[i] += np.random.uniform(-1, 1)
                new_population.append(trial)
            return np.array(new_population)

        population = initialize_population()
        best_individual = population[np.argmin([func(ind) for ind in population])]
        remaining_budget = self.budget - self.population_size

        while remaining_budget > 0:
            new_population = de_hs_step(population, best_individual)
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