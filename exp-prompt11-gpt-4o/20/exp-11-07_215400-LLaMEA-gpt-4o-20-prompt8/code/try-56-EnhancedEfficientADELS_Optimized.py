import numpy as np

class EnhancedEfficientADELS_Optimized:
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
        best_solution = pop[best_idx]
        best_fitness = fitness[best_idx]

        # Pre-calculate random indices for efficiency
        while evals < self.budget:
            idxs = np.random.randint(0, self.population_size, (self.population_size, 3))
            diff_vectors = pop[idxs[:, 1]] - pop[idxs[:, 2]]
            trial_vectors = np.clip(pop[idxs[:, 0]] + self.F * diff_vectors, self.bounds[0], self.bounds[1])

            crossover_points = (np.random.rand(self.population_size, self.dim) < self.CR)
            rand_dim_indices = np.random.randint(0, self.dim, self.population_size)
            np.put_along_axis(crossover_points, rand_dim_indices[:, None], True, axis=1)
            offspring = np.where(crossover_points, trial_vectors, pop)

            # Integrate perturbations more efficiently
            perturb_idx = np.random.rand() < 0.05
            if perturb_idx:
                mutations = np.random.uniform(-0.05, 0.05, self.dim)
                offspring[np.random.randint(0, self.population_size)] += mutations
                np.clip(offspring, self.bounds[0], self.bounds[1], out=offspring)

            offspring_fitness = np.apply_along_axis(func, 1, offspring)
            evals += self.population_size

            # Vectorized condition check
            better_mask = offspring_fitness < fitness
            np.copyto(fitness, offspring_fitness, where=better_mask)
            np.copyto(pop, offspring, where=better_mask[:, None])

            min_fitness_idx = np.argmin(offspring_fitness)
            if offspring_fitness[min_fitness_idx] < best_fitness:
                best_solution = offspring[min_fitness_idx]
                best_fitness = offspring_fitness[min_fitness_idx]

            if evals >= self.budget:
                break

        return best_solution, best_fitness