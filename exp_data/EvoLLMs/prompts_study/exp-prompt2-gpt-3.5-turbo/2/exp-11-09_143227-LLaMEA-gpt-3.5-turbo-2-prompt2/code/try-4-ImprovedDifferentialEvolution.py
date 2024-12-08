import numpy as np

class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, pop_size=20):
        super().__init__(budget, dim, F=0.8, CR=0.9, pop_size=pop_size)

    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        F_adaptive = self.F
        CR_adaptive = self.CR
        
        for _ in range(self.budget):
            new_population = []
            for i, target in enumerate(population):
                mutant = mutate(target, population, F_adaptive)
                trial = crossover(target, mutant, CR_adaptive)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
                
                F_adaptive = max(0.1, min(0.9, F_adaptive + 0.01 * (np.random.rand() - 0.5)))
                CR_adaptive = max(0.1, min(0.9, CR_adaptive + 0.01 * (np.random.rand() - 0.5)))
                
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]