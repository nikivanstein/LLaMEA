import numpy as np

class AdaptiveMemeticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.inertia_weight = 0.7
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5
        self.local_search_rate = 0.1  # fraction of best particles undergoing local search

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        
        global_best_index = np.argmin(fitness)
        global_best = population[global_best_index]
        global_best_fitness = fitness[global_best_index]
        
        while num_evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coefficient * r1 * (personal_best - population) +
                          self.social_coefficient * r2 * (global_best - population))
            population = population + velocities
            population = np.clip(population, self.lb, self.ub)
            
            # Evaluate new solutions
            for i in range(self.population_size):
                current_fitness = func(population[i])
                num_evaluations += 1
                
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < global_best_fitness:
                        global_best = population[i]
                        global_best_fitness = current_fitness
                
                if num_evaluations >= self.budget:
                    break
            
            # Local Search on top-performing particles
            num_local_search = int(self.local_search_rate * self.population_size)
            top_indices = np.argsort(personal_best_fitness)[:num_local_search]
            for i in top_indices:
                if num_evaluations >= self.budget:
                    break
                candidate = personal_best[i] + np.random.normal(0, 0.1, self.dim)
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                num_evaluations += 1
                if candidate_fitness < personal_best_fitness[i]:
                    personal_best[i] = candidate
                    personal_best_fitness[i] = candidate_fitness
                    if candidate_fitness < global_best_fitness:
                        global_best = candidate
                        global_best_fitness = candidate_fitness
        
        return global_best, global_best_fitness