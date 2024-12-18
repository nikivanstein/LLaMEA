import numpy as np

class HADE_RMC_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.success_rate = 0.2
        self.adapt_rate = 0.1

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def mutate(self, population, best_idx, fitness, i):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        lerp_factor = 0.1 + 0.1 * np.sin(np.pi * self.generation / (self.budget / self.pop_size))
        combined = (1 - lerp_factor) * (b - c) + lerp_factor * (best - a)
        improvement_factor = np.exp(-(fitness[i] - np.min(fitness)) / np.abs(np.mean(fitness)))
        mutant = a + self.mutation_factor * combined * improvement_factor
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation, fitness, successful_mutations):
        success_rate = successful_mutations / self.pop_size
        diversity = np.std(fitness) / np.abs(np.mean(fitness))
        if success_rate > self.success_rate:
            self.mutation_factor = min(1.0, self.mutation_factor + self.adapt_rate * diversity)
            self.crossover_probability = min(1.0, self.crossover_probability + self.adapt_rate)
        else:
            self.mutation_factor = max(0.4, self.mutation_factor - self.adapt_rate * diversity)
            self.crossover_probability = max(0.4, self.crossover_probability - self.adapt_rate)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        
        self.generation = 0
        while evaluations < self.budget:
            new_population = np.copy(population)
            successful_mutations = 0
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
                    successful_mutations += 1
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i

            population = new_population
            self.generation += 1
            self.adaptive_strategy(self.generation, fitness, successful_mutations)
        
        return population[best_idx]