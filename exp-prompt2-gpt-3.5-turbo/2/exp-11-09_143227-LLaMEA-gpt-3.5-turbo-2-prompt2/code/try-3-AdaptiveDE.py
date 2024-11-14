import numpy as np

class AdaptiveDE(DifferentialEvolution):
    def __init__(self, budget, dim, F_init=0.8, CR_init=0.9, pop_size=20):
        super().__init__(budget, dim, F_init, CR_init, pop_size)

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
            new_population = []
            for i, target in enumerate(population):
                F = self.F * np.random.uniform(0.5, 1.5)
                CR = self.CR * np.random.uniform(0.5, 1.0)
                mutant = mutate(target, population, F)
                trial = crossover(target, mutant, CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]