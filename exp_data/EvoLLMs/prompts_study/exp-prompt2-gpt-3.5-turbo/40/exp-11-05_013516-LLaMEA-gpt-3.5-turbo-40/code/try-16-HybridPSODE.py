import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iters = 100
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.3
    
    def init_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
    
    def mutate(self, population, target_idx):
        r1, r2, r3 = np.random.choice(len(population), 3, replace=False)
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        return mutant
    
    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() > self.CR:
                trial[i] = mutant[i]
        return trial
    
    def optimize(self, func):
        population = self.init_population()
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.max_iters):
            for i in range(self.population_size):
                target = population[i]
                mutant = self.mutate(population, i)
                trial = self.crossover(target, mutant)
                
                target_fitness = func(target)
                trial_fitness = func(trial)
                
                if trial_fitness < target_fitness:
                    population[i] = trial
                    fitness_values[i] = trial_fitness
                
                if np.sum(fitness_values) >= self.budget:
                    break
        
        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        return best_solution