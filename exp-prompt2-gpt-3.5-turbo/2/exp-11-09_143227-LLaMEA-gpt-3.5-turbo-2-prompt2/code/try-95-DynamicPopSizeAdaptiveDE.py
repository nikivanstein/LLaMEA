import numpy as np

class DynamicPopSizeAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        super().__init__(budget, dim, F, CR, pop_size)

    def __call__(self, func):
        def mutate(x, population, F):
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

            # Dynamic population size adaptation
            if _ % 100 == 0 and self.pop_size < 50:
                self.pop_size += 5
                population = np.vstack([population, np.random.uniform(-5, 5, (5, self.dim))])
                fitness = np.append(fitness, [func(individual) for individual in population[-5:]])

        best_idx = np.argmin(fitness)
        return population[best_idx]