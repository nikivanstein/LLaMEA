import numpy as np

class ImprovedDynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.CR = 0.5  # Crossover rate
        self.F_min = 0.2  # Minimum scaling factor
        self.F_max = 0.8  # Maximum scaling factor

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            F = self.F_min + np.random.rand() * (self.F_max - self.F_min)
            for i in range(self.budget):
                a, b, c = np.random.choice(population, 3, replace=False)
                j_rand = np.random.randint(self.dim)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                
                trial = np.array([mutant[dim] if np.random.rand() < self.CR or dim == j_rand else population[i, dim] for dim in range(self.dim)])
                f_trial = func(trial)
                
                if f_trial < fitness_values[i]:  # Only evaluate fitness if it improves
                    population[i] = trial
                    fitness_values[i] = f_trial
        
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution