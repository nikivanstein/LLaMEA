import numpy as np

class ImprovedDifferentialEvolution(DifferentialEvolution):
    def __call__(self, func):
        def mutate(x, population, F):
            a, b, c = population[np.random.choice(len(population), 3, replace=False)]
            return np.clip(a + F * (b - c), -5, 5)

        def crossover(target, trial, CR):
            mask = np.random.rand(self.dim) < CR
            return np.where(mask, trial, target)

        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        F_min, F_max = 0.2, 0.8
        for _ in range(self.budget):
            new_population = []
            F = F_min + (_ / self.budget) * (F_max - F_min)
            for i, target in enumerate(population):
                mutant = mutate(target, population, F)
                trial = crossover(target, mutant, self.CR)
                new_fitness = func(trial)
                if new_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = new_fitness
                new_population.append(population[i])
            population = np.array(new_population)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]