import numpy as np

class DynamicCompactDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 4 * dim
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8  # dynamic mutation factor for exploration
        self.crossover_rate = 0.8  # dynamic crossover rate for effective recombination
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.success_history = []

    def __call__(self, func):
        evaluations = 0

        for i in range(self.pop_size):
            if evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            evaluations += 1

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)

                F = self.mutation_factor * (1 + 0.2 * np.random.uniform(-0.5, 0.5))  # more adaptive scaling
                mutant_vector = self.population[a] + F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])

                random_index = np.random.randint(self.dim)
                trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.crossover_rate or j == random_index else self.population[i][j] for j in range(self.dim)])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    self.success_history.append((i, trial_fitness))

                if np.random.rand() < 0.15:
                    local_vector = trial_vector + np.random.normal(0, 0.1, self.dim)
                    local_vector = np.clip(local_vector, self.bounds[0], self.bounds[1])
                    local_fitness = func(local_vector)
                    evaluations += 1
                    if local_fitness < trial_fitness:
                        self.population[i] = local_vector
                        self.fitness[i] = local_fitness

            if self.success_history:
                last_success = self.success_history[-1]
                if evaluations % (self.budget // 10) == 0:
                    self.mutation_factor = 0.8 * (1 + 0.1 * np.random.uniform(-1, 1))
                    self.crossover_rate = 0.8 + 0.1 * np.random.uniform(-0.5, 0.5)

        return self.population[np.argmin(self.fitness)], self.fitness[np.argmin(self.fitness)]