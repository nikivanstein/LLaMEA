import numpy as np

class DynamicPopulationDE(DifferentialEvolution):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        super().__init__(budget, dim, F, CR, pop_size)
        self.min_pop_size = 10
        self.max_pop_size = 30

    def __call__(self, func):
        def adapt_population_size(iteration):
            return max(self.min_pop_size, min(self.max_pop_size, int(self.pop_size * (1 + 0.1 * np.sin(iteration * np.pi / self.budget)))))

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for t in range(self.budget):
            self.pop_size = adapt_population_size(t)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, self.F)
                trial = crossover(target, mutant, self.CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]