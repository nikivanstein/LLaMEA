import numpy as np

class HADE_RMC_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_pop_size = min(100, self.budget // self.dim)
        self.max_pop_size = 200  # Dynamic population size
        self.mutation_factor = 0.6  # Adjusted for better exploration
        self.crossover_probability = 0.7  # Adjusted for better exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.learning_rate = 0.1  # Adaptive learning rate

    def initialize_population(self, pop_size):
        return np.random.uniform(self.lower_bound, self.upper_bound, (pop_size, self.dim))

    def mutate(self, population, best_idx, fitness, i):
        idxs = np.random.choice(len(population), 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        improvement_factor = np.exp(-(fitness[i] - np.min(fitness)) / np.abs(np.mean(fitness)))
        mutant = a + self.mutation_factor * (b - c + self.learning_rate * (best - a)) * improvement_factor
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation, fitness):
        diversity = np.std(fitness) / np.abs(np.mean(fitness))
        self.mutation_factor = 0.5 + 0.3 * np.cos(np.pi * generation / (self.budget / self.initial_pop_size)) * diversity
        self.crossover_probability = 0.6 + 0.4 * np.sin(np.pi * generation / (self.budget / self.initial_pop_size))
        if diversity < 0.1:
            self.learning_rate *= 1.05  # Increase learning rate if diversity is low
        self.population_size = min(self.max_pop_size, self.initial_pop_size + int(diversity * 100))

    def __call__(self, func):
        population = self.initialize_population(self.initial_pop_size)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size
        best_idx = np.argmin(fitness)

        self.generation = 0
        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(len(population)):
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

            population = self.initialize_population(self.population_size)
            fitness = np.array([func(ind) for ind in population])  # Re-evaluate fitness for new population size
            self.generation += 1
            self.adaptive_strategy(self.generation, fitness)

        return population[np.argmin(fitness)]