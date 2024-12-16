import numpy as np

class HADE_RMC_Adapt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
    
    def mutate(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        lerp_factor = 0.1 + 0.4 * (1 - self.evaluations / self.budget)
        combined = (1 - lerp_factor) * (b - c) + lerp_factor * (best - a)
        mutant = a + self.mutation_factor * combined
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation):
        self.mutation_factor = 0.6 + 0.4 * np.cos(np.pi * generation / (self.budget / self.pop_size))
        self.crossover_probability = 0.4 + 0.6 * np.sin(np.pi * generation / (self.budget / self.pop_size))
        self.pop_size = max(20, int(self.pop_size * (0.9 + 0.1 * np.random.rand())))
        
    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        
        generation = 0
        while self.evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                
                mutant = self.mutate(population, best_idx)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                self.evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i

            population = new_population
            generation += 1
            self.adaptive_strategy(generation)
        
        return population[best_idx]