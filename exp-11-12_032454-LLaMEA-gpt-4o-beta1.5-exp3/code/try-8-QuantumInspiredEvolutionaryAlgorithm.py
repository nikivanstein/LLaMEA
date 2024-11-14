import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))
        self.mutation_rate = 0.1
        self.rotation_angle = 0.05

    def __call__(self, func):
        # Initialize quantum bits (position) and make copies for personal/global best
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        
        global_best_index = np.argmin(fitness)
        global_best = population[global_best_index]
        global_best_fitness = fitness[global_best_index]
        
        while num_evaluations < self.budget:
            # Quantum rotation update
            for i in range(self.population_size):
                rotation_vector = np.random.uniform(-self.rotation_angle, self.rotation_angle, self.dim)
                population[i] = population[i] + rotation_vector * np.sign(global_best - population[i])
                population[i] = np.clip(population[i], self.lb, self.ub)
            
            # Evaluate new solutions and update personal/global bests
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
            
            # Adaptive mutation to maintain diversity
            mutation_strength = self.mutation_rate * (1 - num_evaluations / self.budget)
            for i in range(self.population_size):
                if np.random.rand() < mutation_strength:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    candidate = personal_best[i] + mutation_vector
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