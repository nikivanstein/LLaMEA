import numpy as np

class EnhancedHybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(80, budget // 4)
        self.mutation_factor = np.random.uniform(0.5, 1.0, self.population_size)
        self.crossover_prob = np.random.uniform(0.5, 0.9, self.population_size)
        self.local_search_prob = 0.3
        self.tournament_size = 2
        self.elitism_rate = 0.1
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(self.fitness)[:elite_count]
            next_population = self.population[elite_indices]
            next_fitness = self.fitness[elite_indices]

            while len(next_population) < self.population_size:
                i = np.random.choice(range(self.population_size))
                if self.evaluations >= self.budget:
                    break

                indices = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(indices, self.tournament_size, replace=False)
                best_idx = min(chosen, key=lambda idx: self.fitness[idx])
                a, b, c = self.population[best_idx], self.population[np.random.choice(indices)], self.population[np.random.choice(indices)]
                mutant = np.clip(a + self.mutation_factor[i] * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_prob[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    next_population = np.vstack((next_population, trial))
                    next_fitness = np.append(next_fitness, trial_fitness)
                else:
                    next_population = np.vstack((next_population, self.population[i]))
                    next_fitness = np.append(next_fitness, self.fitness[i])

                if np.random.rand() < self.local_search_prob:
                    self.adaptive_local_search(i, func, next_population, next_fitness)

            self.population, self.fitness = next_population[:self.population_size], next_fitness[:self.population_size]

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def adaptive_local_search(self, index, func, pop, fit):
        step_size = 0.03 * (self.upper_bound - self.lower_bound)
        for _ in range(2):
            if self.evaluations >= self.budget:
                break

            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            neighbor = np.clip(pop[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1

            if neighbor_fitness < fit[index]:
                pop[index] = neighbor
                fit[index] = neighbor_fitness