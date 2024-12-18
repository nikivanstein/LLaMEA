import numpy as np

class Enhanced_HADE_RMC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(100, self.budget // self.dim)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.successful_mutations = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def mutate(self, population, best_idx):
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        a, b, c = population[idxs]
        best = population[best_idx]
        diversity_factor = np.std(population, axis=0)
        combined = (b - c) * diversity_factor + (best - a) * (1 - diversity_factor)
        mutant = a + self.mutation_factor * combined
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adaptive_strategy(self, generation, fitness):
        diversity = np.std(fitness) / (np.abs(np.mean(fitness)) + 1e-12)
        success_rate = np.mean(self.successful_mutations[-10:]) if len(self.successful_mutations) >= 10 else 0.5
        self.mutation_factor = 0.4 + 0.6 * np.cos(np.pi * generation / (self.budget / self.pop_size)) * diversity * success_rate
        self.crossover_probability = 0.5 + 0.5 * np.sin(np.pi * generation / (self.budget / self.pop_size))
        
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
                
                mutant = self.mutate(population, best_idx)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    self.successful_mutations.append(1)
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                else:
                    self.successful_mutations.append(0)

            population = new_population
            self.generation += 1
            self.adaptive_strategy(self.generation, fitness)
        
        return population[best_idx]