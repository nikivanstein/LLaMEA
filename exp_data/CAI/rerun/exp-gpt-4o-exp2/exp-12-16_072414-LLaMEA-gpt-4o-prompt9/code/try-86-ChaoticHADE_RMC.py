import numpy as np

class ChaoticHADE_RMC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.logistic_map_r = 3.99
        self.z = 0.7

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def logistic_map(self):
        self.z = self.logistic_map_r * self.z * (1 - self.z)
        return self.z

    def mutate(self, population, best_idx, fitness, i):
        fitness_scaled = 1.0 / (1.0 + fitness - np.min(fitness))
        idxs = np.random.choice(self.pop_size, 3, replace=False, p=fitness_scaled/fitness_scaled.sum())
        a, b, c = population[idxs]
        best = population[best_idx]
        adaptive_factor = 0.2 + 0.3 * self.logistic_map()
        combined = (1 - adaptive_factor) * (b - c) + adaptive_factor * (best - a)
        improvement_factor = np.exp(-(fitness[i] - np.min(fitness)) / np.abs(np.mean(fitness)))
        sinusoidal_factor = 0.02 * np.sin(self.generation * np.pi / (self.budget / self.pop_size))
        diversity_term = np.cos(np.pi * (a - best) @ (b - c)) 
        diversity_scale = np.std(population) / np.mean(np.abs(population))
        mutant = a + (self.mutation_factor + 0.02 * (self.generation / (self.budget / self.pop_size)) + sinusoidal_factor + 0.01 * diversity_term) * combined * improvement_factor * diversity_scale
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation, fitness):
        diversity = np.std(fitness) / np.abs(np.mean(fitness))
        self.mutation_factor = 0.5 + (0.5 * np.cos(np.pi * generation / (self.budget / self.pop_size)) + diversity) * 0.8
        self.crossover_probability = 0.4 + 0.6 * self.logistic_map()

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        
        self.generation = 0
        while evaluations < self.budget:
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
            self.adaptive_strategy(self.generation, fitness)
        
        return population[best_idx]