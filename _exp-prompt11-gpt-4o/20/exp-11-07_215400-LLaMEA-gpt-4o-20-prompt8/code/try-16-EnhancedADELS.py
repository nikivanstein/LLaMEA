import numpy as np

class EnhancedADELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5 * dim, 20)
        self.F = 0.7  # Differential weight
        self.CR = 0.8  # Crossover probability

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            new_pop = np.empty_like(pop)
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[idxs]
                trial_vector = np.clip(x1 + self.F * (x2 - x3), self.bounds[0], self.bounds[1])

                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                offspring = np.where(crossover_mask, trial_vector, pop[i])
                if np.random.rand() < 0.05:  # Reduced perturbation probability for efficiency
                    perturbation = np.random.uniform(-0.05, 0.05, self.dim)
                    offspring = np.clip(offspring + perturbation, self.bounds[0], self.bounds[1])
                
                offspring_fitness = func(offspring)
                evals += 1

                if offspring_fitness < fitness[i]:
                    new_pop[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_solution = offspring.copy()
                        best_fitness = offspring_fitness
                else:
                    new_pop[i] = pop[i]

                if evals >= self.budget:
                    return best_solution, best_fitness

            pop = new_pop

        return best_solution, best_fitness