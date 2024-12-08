import numpy as np

class DynamicEnsembleDE:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.strategies = ['rand-to-best/1/bin', 'best/2/bin', 'rand/1/exp']

    def __call__(self, func):
        def mutate(x, population, F, strategy):
            if strategy == 'rand-to-best/1/bin':
                a, b, c, d = population[np.random.choice(len(population), 4, replace=False)]
                return np.clip(a + F * (b - c) + F * (d - x), -5, 5)
            elif strategy == 'best/2/bin':
                best_idx = np.argmin(fitness)
                best = population[best_idx]
                a, b, c = population[np.random.choice(len(population), 3, replace=False)]
                return np.clip(best + F * (a - b) + F * (c - x), -5, 5)
            elif strategy == 'rand/1/exp':
                a, b, c = population[np.random.choice(len(population), 3, replace=False)]
                return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)  # Adapt F over time
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)  # Adapt CR with sinusoidal function
            strategy = np.random.choice(self.strategies)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F, strategy)
                trial = crossover(target, mutant, adapt_CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]