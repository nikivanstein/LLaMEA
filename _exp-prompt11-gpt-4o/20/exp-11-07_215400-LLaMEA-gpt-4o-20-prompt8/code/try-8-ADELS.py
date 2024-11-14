import numpy as np

class ADELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 10 * dim
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        evals = self.population_size
        
        while evals < self.budget:
            new_pop = np.copy(pop)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                trial_vector = x1 + self.F * (x2 - x3)
                # Enforce bounds
                trial_vector = np.clip(trial_vector, self.bounds[0], self.bounds[1])

                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                if not crossover_mask.any():
                    crossover_mask[np.random.randint(0, self.dim)] = True
                offspring = np.where(crossover_mask, trial_vector, pop[i])
                
                # Local search: small random perturbation
                if np.random.rand() < 0.1:
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                    offspring = np.clip(offspring + perturbation, self.bounds[0], self.bounds[1])
                
                # Evaluate new solution
                offspring_fitness = func(offspring)
                evals += 1
                
                # Selection
                if offspring_fitness < fitness[i]:
                    new_pop[i] = offspring
                    fitness[i] = offspring_fitness

                if evals >= self.budget:
                    break
            
            pop = new_pop
        
        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]