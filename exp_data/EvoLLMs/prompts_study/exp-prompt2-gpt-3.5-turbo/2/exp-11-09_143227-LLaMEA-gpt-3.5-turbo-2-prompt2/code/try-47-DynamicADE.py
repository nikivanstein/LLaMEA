import numpy as np

class DynamicADE:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

    def __call__(self, func):
        def mutate(x, population, F, F_min=0.2, F_max=0.8):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            F = F_min + (F_max - F_min) * np.random.rand()  # Dynamic F within range
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR_min=0.1, CR_max=0.9):
            CR = CR_min + (CR_max - CR_min) * np.random.rand()  # Dynamic CR within range
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
            adapt_F = self.F * (1.0 - _ / self.budget)
            adapt_CR = self.CR + 0.1 * np.sin(0.9 * np.pi * _ / self.budget)
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, adapt_F)
                trial = crossover(target, mutant)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

        best_idx = np.argmin(fitness)
        return population[best_idx]