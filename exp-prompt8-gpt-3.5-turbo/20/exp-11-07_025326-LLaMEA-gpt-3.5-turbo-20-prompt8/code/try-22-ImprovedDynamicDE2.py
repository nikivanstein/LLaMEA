import numpy as np

class ImprovedDynamicDE2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            F = np.random.uniform(0.2, 0.8)  # Optimized random F generation
            for i in range(self.budget):
                idx = np.delete(np.arange(self.budget), i)
                a, b, c = population[np.random.choice(idx, 3, replace=False)]
                j_rand = np.random.randint(self.dim)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                
                trial = np.where(np.random.rand(self.dim) < F, mutant, population[i])  # Use optimized F for crossover
                f_trial = func(trial)
                
                if f_trial < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = f_trial
        
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution