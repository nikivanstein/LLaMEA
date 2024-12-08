import numpy as np

class HybridGeneticDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(120, budget // 3)
        self.mutation_factor = 0.9
        self.crossover_prob = 0.9
        self.elitism_rate = 0.2
        self.local_search_prob = 0.3
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.chaos_sequence = np.random.rand(self.population_size)

    def __call__(self, func):
        self.fitness = np.apply_along_axis(func, 1, self.population)
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            next_population = self.population.copy()
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argpartition(self.fitness, elite_count)[:elite_count]

            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                if i in elite_indices:
                    next_population[i] = self.population[i]
                    continue

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    next_population[i] = trial
                    self.fitness[i] = trial_fitness

                if np.random.rand() < self.local_search_prob:
                    self.adaptive_chaotic_perturbation(i, func)

            self.population = next_population

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def adaptive_chaotic_perturbation(self, index, func):
        chaos_factor = 0.2 + 0.1 * np.random.rand()
        for _ in range(2):
            if self.evaluations >= self.budget:
                break

            self.chaos_sequence[index] = 4 * self.chaos_sequence[index] * (1 - self.chaos_sequence[index])
            step_size = chaos_factor * (self.upper_bound - self.lower_bound) * (0.5 - self.chaos_sequence[index])
            perturbation = np.random.normal(0, np.abs(step_size), self.dim)
            neighbor = np.clip(self.population[index] + perturbation, self.lower_bound, self.upper_bound)
            neighbor_fitness = func(neighbor)
            self.evaluations += 1
            
            if neighbor_fitness < self.fitness[index]:
                self.population[index] = neighbor
                self.fitness[index] = neighbor_fitness