import numpy as np

class ImprovedDynamicDE_Variant:
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
            random_indices = np.random.choice(self.budget, 3, replace=False)
            a, b, c = population[random_indices]
            j_rand = np.random.randint(self.dim)
            mutant = np.clip(a + F * (b - c), -5.0, 5.0)
            
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population)
            f_trial = np.array([func(individual) for individual in trial])
            
            improved_fitness_mask = f_trial < fitness_values
            population[improved_fitness_mask] = trial[improved_fitness_mask]
            fitness_values[improved_fitness_mask] = f_trial[improved_fitness_mask]
        
        best_index = np.argmin(fitness_values)
        best_solution = population[best_index]
        return best_solution