import numpy as np

class EfficientADELS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = max(5 * dim, 20)
        self.F = 0.7
        self.CR = 0.8

    def __call__(self, func):
        pop = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        evals = self.population_size
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx].copy()
        best_fitness = fitness[best_idx]

        while evals < self.budget:
            idxs = np.random.choice(self.population_size, (self.population_size, 3))
            trial_vectors = np.clip(pop[idxs[:, 0]] + self.F * (pop[idxs[:, 1]] - pop[idxs[:, 2]]), self.bounds[0], self.bounds[1])

            crossover_masks = np.random.rand(self.population_size, self.dim) < self.CR
            crossover_masks[np.arange(self.population_size), np.random.randint(0, self.dim, self.population_size)] = True
            offspring = np.where(crossover_masks, trial_vectors, pop)

            mutations = np.random.uniform(-0.05, 0.05, (self.population_size, self.dim))
            perturb_chance = np.random.rand(self.population_size) < 0.05
            offspring = np.clip(offspring + perturb_chance[:, None] * mutations, self.bounds[0], self.bounds[1])

            offspring_fitness = np.apply_along_axis(func, 1, offspring)
            evals += self.population_size

            improvement_mask = offspring_fitness < fitness
            fitness[improvement_mask] = offspring_fitness[improvement_mask]
            pop[improvement_mask] = offspring[improvement_mask]

            if offspring_fitness.min() < best_fitness:
                best_idx = np.argmin(offspring_fitness)
                best_solution = offspring[best_idx].copy()
                best_fitness = offspring_fitness[best_idx]

            if evals >= self.budget:
                break

        return best_solution, best_fitness