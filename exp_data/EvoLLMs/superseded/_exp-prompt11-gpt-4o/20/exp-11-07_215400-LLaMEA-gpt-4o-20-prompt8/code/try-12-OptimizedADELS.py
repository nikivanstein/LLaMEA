import numpy as np

class OptimizedADELS:
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
            indices = np.random.choice(self.population_size, (self.population_size, 3), replace=True)
            new_pop = np.empty_like(pop)
            for i, (idx1, idx2, idx3) in enumerate(indices):
                x1, x2, x3 = pop[idx1], pop[idx2], pop[idx3]
                trial_vector = np.clip(x1 + self.F * (x2 - x3), self.bounds[0], self.bounds[1])

                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                
                offspring = np.where(crossover_mask, trial_vector, pop[i])

                if np.random.rand() < 0.1:
                    perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                    offspring = np.clip(offspring + perturbation, self.bounds[0], self.bounds[1])
                
                offspring_fitness = func(offspring)
                evals += 1

                if offspring_fitness < fitness[i]:
                    new_pop[i] = offspring
                    fitness[i] = offspring_fitness
                else:
                    new_pop[i] = pop[i]

                if evals >= self.budget:
                    break

            pop = new_pop

        best_idx = np.argmin(fitness)
        return pop[best_idx], fitness[best_idx]