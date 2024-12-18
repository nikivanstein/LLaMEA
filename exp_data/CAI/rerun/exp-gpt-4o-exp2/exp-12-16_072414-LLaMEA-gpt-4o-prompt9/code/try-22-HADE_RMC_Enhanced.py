import numpy as np

class HADE_RMC_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.mutation_factor = 0.7
        self.crossover_probability = 0.8
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
    
    def dynamic_population_scaling(self, generation):
        scale_factor = 1 + 0.1 * np.sin(np.pi * generation / (self.budget / self.pop_size))
        self.pop_size = int(self.pop_size * scale_factor)
        self.pop_size = min(self.pop_size, self.budget // self.dim)
    
    def chaotic_mutation_operator(self, generation):
        return 0.7 + 0.5 * np.abs(np.sin(generation))

    def mutate(self, population, best_idx, fitness, i, generation):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        chaotic_factor = self.chaotic_mutation_operator(generation)
        combined = chaotic_factor * (b - c) + (1 - chaotic_factor) * (best - a)
        improvement_factor = np.exp(-(fitness[i] - np.min(fitness)) / (1 + np.std(fitness)))
        mutant = a + self.mutation_factor * combined * improvement_factor
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation, fitness):
        diversity = np.std(fitness) / (1 + np.abs(np.mean(fitness)))
        self.mutation_factor = 0.5 + 0.5 * np.cos(np.pi * generation / (self.budget / self.pop_size)) * diversity
        self.crossover_probability = 0.6 + 0.4 * np.sin(np.pi * generation / (self.budget / self.pop_size))
        
    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        
        self.generation = 0
        while evaluations < self.budget:
            self.dynamic_population_scaling(self.generation)
            new_population = np.copy(population)
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break
                
                mutant = self.mutate(population, best_idx, fitness, i, self.generation)
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