import numpy as np

class ImprovedDE:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c, d = population[np.random.choice(len(population), 4, replace=False)]
            return np.clip(x + F * (a - b + c - d), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            dynamic_F = self.F * np.exp(-_ / self.budget)  # Dynamic scaling factor for F
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)  # Adapt CR with a sinusoidal function
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, dynamic_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]