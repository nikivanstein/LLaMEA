import numpy as np

class RefinedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, F=0.8, CR=0.9, pop_size=20, local_search_prob=0.1):
        super().__init__(budget, dim, F, CR, pop_size)
        self.local_search_prob = local_search_prob

    def local_search(self, population, func):
        for i in range(len(population)):
            candidate = population[i].copy()
            for j in range(self.dim):
                if np.random.rand() < self.local_search_prob:
                    candidate[j] = np.clip(candidate[j] + np.random.uniform(-0.1, 0.1), -5, 5)
            if func(candidate) < func(population[i]):
                population[i] = candidate

    def __call__(self, func):
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
            
            self.local_search(population, func)
        
        best_idx = np.argmin(fitness)
        return population[best_idx]