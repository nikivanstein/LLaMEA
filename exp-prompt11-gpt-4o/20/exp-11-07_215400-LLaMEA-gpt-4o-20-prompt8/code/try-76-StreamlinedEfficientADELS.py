import numpy as np

class StreamlinedEfficientADELS:
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
        
        rng = np.random.default_rng()

        while evals < self.budget:
            # Vectorized index selection and crossover
            idxs = rng.integers(self.population_size, size=(self.population_size, 3))
            diff_vectors = pop[idxs[:, 1]] - pop[idxs[:, 2]]
            trial_vectors = np.clip(pop[idxs[:, 0]] + self.F * diff_vectors, self.bounds[0], self.bounds[1])

            crossover_mask = rng.random((self.population_size, self.dim)) < self.CR
            random_dimen = rng.integers(0, self.dim, self.population_size)
            crossover_mask[np.arange(self.population_size), random_dimen] = True
            offspring = np.where(crossover_mask, trial_vectors, pop)

            # Perturbation reduced to a simple vector addition
            if rng.random() < 0.05:
                perturb_idx = rng.integers(self.population_size)
                mutations = rng.uniform(-0.05, 0.05, self.dim)
                offspring[perturb_idx] = np.clip(offspring[perturb_idx] + mutations, self.bounds[0], self.bounds[1])

            offspring_fitness = np.apply_along_axis(func, 1, offspring)
            evals += self.population_size

            better_mask = offspring_fitness < fitness
            fitness = np.where(better_mask, offspring_fitness, fitness)
            pop = np.where(better_mask[:, None], offspring, pop)

            min_fitness_idx = np.argmin(offspring_fitness)
            if offspring_fitness[min_fitness_idx] < best_fitness:
                best_solution = offspring[min_fitness_idx].copy()
                best_fitness = offspring_fitness[min_fitness_idx]

            if evals >= self.budget:
                break

        return best_solution, best_fitness