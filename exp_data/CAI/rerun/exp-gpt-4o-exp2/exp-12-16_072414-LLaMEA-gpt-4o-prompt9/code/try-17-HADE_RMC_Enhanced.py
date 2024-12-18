import numpy as np

class HADE_RMC_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.initial_pop_size = self.pop_size
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.resizing_factor = 0.5

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def mutate(self, population, best_idx, fitness, i):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        combined = b - c + best - a
        improvement_factor = np.log1p(np.max(fitness) - fitness[i]) / np.log1p(np.max(fitness))
        mutant = a + self.mutation_factor * combined * improvement_factor
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def dynamic_parameters(self, generation, fitness):
        diversity = np.std(fitness) / np.abs(np.mean(fitness))
        self.mutation_factor = 0.4 + 0.6 * np.cos(np.pi * generation / (self.budget / self.pop_size)) * diversity
        self.crossover_probability = 0.5 + 0.4 * np.cos(np.pi * generation / (self.budget / self.pop_size))
        if generation % 10 == 0:
            self.pop_size = max(20, int(self.initial_pop_size * (1 - self.resizing_factor * generation / (self.budget / self.pop_size))))

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)

        self.generation = 0
        while evaluations < self.budget:
            self.dynamic_parameters(self.generation, fitness)
            new_population = np.copy(population)
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                mutant = self.mutate(population, best_idx, fitness, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i

            population = new_population
            self.generation += 1

        return population[best_idx]