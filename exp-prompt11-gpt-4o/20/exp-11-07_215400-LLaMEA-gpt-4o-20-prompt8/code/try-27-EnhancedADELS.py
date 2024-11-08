import numpy as np

class EnhancedADELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5 * dim, 20)
        self.F = 0.7
        self.CR = 0.8

    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            indices = np.arange(self.population_size)
            np.random.shuffle(indices)
            
            for i in range(self.population_size):
                idxs = indices[i:i+3]
                if len(idxs) < 3:
                    idxs = np.concatenate((idxs, indices[:3-len(idxs)]))
                x1, x2, x3 = pop[idxs]
                trial_vector = x1 + self.F * (x2 - x3)
                np.clip(trial_vector, self.bounds[0], self.bounds[1], out=trial_vector)

                crossover_mask = np.random.rand(self.dim) < self.CR
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                offspring = np.where(crossover_mask, trial_vector, pop[i])
                if np.random.rand() < 0.05:
                    offspring += np.random.uniform(-0.05, 0.05, self.dim)
                    np.clip(offspring, self.bounds[0], self.bounds[1], out=offspring)

                offspring_fitness = func(offspring)
                evals += 1

                if offspring_fitness < fitness[i]:
                    pop[i] = offspring
                    fitness[i] = offspring_fitness
                    if offspring_fitness < best_fitness:
                        best_solution = offspring.copy()
                        best_fitness = offspring_fitness

                if evals >= self.budget:
                    return best_solution, best_fitness

        return best_solution, best_fitness