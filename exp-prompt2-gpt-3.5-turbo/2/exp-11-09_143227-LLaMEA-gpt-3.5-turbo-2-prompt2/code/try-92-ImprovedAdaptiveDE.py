import numpy as np

class ImprovedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, cauchy_scale=0.1):
        super().__init__(budget, dim, F, CR, pop_size)
        self.cauchy_scale = cauchy_scale

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            cauchy_dist = np.random.standard_cauchy(self.dim) * self.cauchy_scale
            return np.clip(x + F * (b - c) + cauchy_dist, -5, 5)
        
        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]