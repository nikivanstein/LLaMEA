import numpy as np

class ADE_LSB:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.5
        self.CR = 0.9
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None

    def __call__(self, func):
        evaluations = 0
        fitness = np.array([func(ind) for ind in self.pop])
        evaluations += self.population_size
        self.best_solution = self.pop[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Adaptive parameter control
                F_adaptive = np.random.rand() * 0.9 + 0.1  # Updated differential scaling factor range
                CR_adaptive = np.random.rand() * 0.5 + 0.4  # Adjusted CR_adaptive for better exploration

                # Mutation and crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                f_scale = np.random.rand()  # Removed fitness scaling for cleaner mutation
                mutant_vector = self.pop[a] + f_scale * F_adaptive * (self.pop[b] - self.pop[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.copy(self.pop[i])
                cross_points = np.random.rand(self.dim) < CR_adaptive
                trial_vector[cross_points] = mutant_vector[cross_points]

                # Local search boosting
                if np.random.rand() < 0.15 * (1 - evaluations / self.budget):  # Dynamic local search probability
                    trial_vector = self.local_search(trial_vector, func)

                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    self.pop[i] = trial_vector
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        self.best_solution = trial_vector

        return self.best_solution

    def local_search(self, vector, func):
        # Simple local search using a small random perturbation
        perturbation = np.random.normal(0, 0.02, self.dim)  # Adjusted perturbation distribution
        new_vector = vector + perturbation
        new_vector = np.clip(new_vector, self.lower_bound, self.upper_bound)
        if func(new_vector) < func(vector):
            return new_vector
        return vector