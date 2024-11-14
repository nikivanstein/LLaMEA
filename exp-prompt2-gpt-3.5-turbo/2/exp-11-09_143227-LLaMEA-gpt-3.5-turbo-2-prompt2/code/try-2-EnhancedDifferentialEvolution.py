import numpy as np

class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, pop_size=20):
        super().__init__(budget, dim, pop_size=pop_size)

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
            diversity = np.std(population, axis=0)

            F = np.clip(0.5 + 0.3 * np.mean(diversity), 0, 1)
            CR = np.clip(0.5 + 0.3 * np.mean(diversity), 0, 1)

            for i, target in enumerate(population):
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