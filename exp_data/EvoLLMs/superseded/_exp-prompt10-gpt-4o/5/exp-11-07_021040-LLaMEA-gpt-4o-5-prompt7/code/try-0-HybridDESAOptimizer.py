import numpy as np

class HybridDESAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.8  # Differential Evolution parameter
        self.CR = 0.9  # Crossover probability
        self.initial_temp = 100
        self.cooling_rate = 0.99

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Differential Evolution
                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate the trial vector
                trial_fitness = func(trial)
                evaluations += 1
                
                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    
                # Simulated Annealing acceptance
                temp = self.initial_temp * (self.cooling_rate ** evaluations)
                if trial_fitness < best_fitness or np.exp((best_fitness - trial_fitness) / temp) > np.random.rand():
                    best = trial.copy()
                    best_fitness = trial_fitness
            
            population = new_population
            
            # Early exit if evaluations exceed budget
            if evaluations >= self.budget:
                break
        
        return best, best_fitness