import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size

    def __call__(self, func):
        def mutate(x, population, F):
            scaling_factor = np.random.uniform(0.5, 1.0)  # Introduce adaptive scaling factor
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + scaling_factor * F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        for _ in range(self.budget):
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