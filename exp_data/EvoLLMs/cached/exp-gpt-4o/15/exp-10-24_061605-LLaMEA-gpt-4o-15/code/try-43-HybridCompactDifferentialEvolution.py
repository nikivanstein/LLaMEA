import numpy as np

class HybridCompactDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.9
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def __call__(self, func):
        evaluations = 0

        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            evaluations += 1

        best_idx = np.argmin(self.fitness)
        best_solution = self.population[best_idx].copy()

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                F = self.mutation_factor * np.random.uniform(0.5, 1.0)
                mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness

                if np.random.rand() < 0.15:
                    local_vector = trial_vector + np.random.normal(0, 0.05, self.dim)
                    local_vector = np.clip(local_vector, self.bounds[0], self.bounds[1])
                    local_fitness = func(local_vector)
                    evaluations += 1
                    if local_fitness < trial_fitness:
                        self.population[i] = local_vector
                        self.fitness[i] = local_fitness

            current_best_idx = np.argmin(self.fitness)
            if self.fitness[current_best_idx] < self.fitness[best_idx]:
                best_idx = current_best_idx
                best_solution = self.population[best_idx].copy()

        return best_solution, self.fitness[best_idx]