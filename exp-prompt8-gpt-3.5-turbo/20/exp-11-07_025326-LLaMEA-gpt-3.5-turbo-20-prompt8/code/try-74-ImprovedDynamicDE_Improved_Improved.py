import numpy as np

class ImprovedDynamicDE_Improved_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (int(0.8*self.budget), self.dim))  # Reduced population initialization size
        fitness_values = np.array([func(individual) for individual in population])
        
        for t in range(self.budget):
            if t % 5 == 0:  # Update population every 5 iterations
                for i in range(len(population)):
                    F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
                    idx = np.delete(np.arange(len(population)), i)
                    a, b, c = population[np.random.choice(idx, 3, replace=False)]
                    j_rand = np.random.randint(self.dim)
                    mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                    
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                    f_trial = func(trial)
                    
                    if f_trial < fitness_values[i]:
                        population[i] = trial
                        fitness_values[i] = f_trial
        
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution