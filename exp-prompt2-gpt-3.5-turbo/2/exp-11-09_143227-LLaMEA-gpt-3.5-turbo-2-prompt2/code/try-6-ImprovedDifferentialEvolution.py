import numpy as np

class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, selection_pressure=0.7):
        super().__init__(budget, dim, F, CR, pop_size)
        self.selection_pressure = selection_pressure

    def __call__(self, func):
        def mutate(x, population, F, best_idx):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            if np.random.rand() < self.selection_pressure:
                target = population[best_idx]
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            new_population = []
            best_idx = np.argmin(fitness)
            for i, target in enumerate(population):
                mutant = mutate(target, population, self.F, best_idx)
                trial = crossover(target, mutant, self.CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]