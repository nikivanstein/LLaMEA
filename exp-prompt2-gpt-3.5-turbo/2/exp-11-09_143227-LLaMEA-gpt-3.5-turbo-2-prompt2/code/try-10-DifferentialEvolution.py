import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, F_min=0.4, F_max=1.0, F_decay=0.95):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_min = F_min
        self.F_max = F_max
        self.F_decay = F_decay

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
                self.F = max(self.F_min, min(self.F_max, self.F * self.F_decay))
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