import numpy as np

class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, scaling_factor=0.5):
        super().__init__(budget, dim, F, CR, pop_size)
        self.scaling_factor = scaling_factor

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * (b - c) * self.scaling_factor, -5, 5)

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