import numpy as np

class DynamicControlDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.9
        self.F = 0.5

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            target = population[i]
            idxs = np.arange(self.budget)
            np.random.shuffle(idxs)
            a, b, c = population[np.random.choice(idxs[:3], 3, replace=False)]
            mutant = a + self.F * (b - c)
            
            # Dynamic control of crossover rate based on population diversity
            avg_distance = np.mean(np.linalg.norm(population - population.mean(axis=0), axis=1))
            self.CR = 0.9 if avg_distance < 1.0 else 0.5
            
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, target)
            
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                
                # Dynamic control of mutation rate based on fitness improvement
                self.F = 0.9 if trial_fitness < fitness.mean() else 0.5
        
        best_idx = np.argmin(fitness)
        return population[best_idx]