import numpy as np

class AdaptiveDE(DifferentialEvolution):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20):
        super().__init__(budget, dim, F, CR, pop_size)
        self.min_pop_size = 10
        self.max_pop_size = 50

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
                mutant = mutate(target, population, self.F)
                trial = crossover(target, mutant, self.CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)

            # Dynamic population size adaptation
            if np.random.rand() < 0.1 and self.pop_size > self.min_pop_size:
                self.pop_size = max(self.min_pop_size, int(self.pop_size * 0.9))
                population = np.concatenate([population, np.random.uniform(-5, 5, (self.pop_size - len(population), self.dim))])
                fitness = np.concatenate([fitness, np.array([func(individual) for individual in population[len(fitness):]])])
            elif np.random.rand() < 0.1 and self.pop_size < self.max_pop_size:
                self.pop_size = min(self.max_pop_size, int(self.pop_size * 1.1))
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]
        
        best_idx = np.argmin(fitness)
        return population[best_idx]