import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.cr = 0.5
        self.f = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(population):
            return np.array([func(solution) for solution in population])

        def differential_evolution(population, fitness):
            for i in range(self.budget):
                target = population[i]
                indices = list(range(self.budget))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.f * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, target)
                
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            return population, fitness

        population = initialize_population()
        fitness = get_fitness(population)
        
        for _ in range(self.budget // 10):
            population, fitness = differential_evolution(population, fitness)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]