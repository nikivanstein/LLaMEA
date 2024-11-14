import numpy as np

class AdaptiveSwarmQuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (8 * dim)))  # adjusted heuristic for population size
        self.F_min, self.F_max = 0.4, 0.9  # range for adaptive scaling factor
        self.CR_min, self.CR_max = 0.6, 1.0  # range for adaptive crossover probability
        self.alpha = 0.1  # learning factor for dynamic adjustments
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        quantum_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Adaptive parameter adjustment
                F = np.random.uniform(self.F_min, self.F_max)
                CR = np.random.uniform(self.CR_min, self.CR_max)
                
                # Mutation using Swarm-inspired differential
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutated_vector = population[a] + F * (population[b] - population[c])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                
                # Quantum crossover
                crossover_mask = np.random.rand(self.dim) < CR
                offspring = np.where(crossover_mask, mutated_vector, population[i])
                
                # Superposition-based enhancement with swarm influence
                quantum_selection = np.random.rand(self.dim) < quantum_population[i]
                swarm_influence = np.random.uniform(self.lb, self.ub, self.dim)
                offspring = np.where(quantum_selection, offspring, 0.5 * (best_individual + swarm_influence))
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1
                
                # Selection and dynamic adjustment
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    quantum_population[i] = np.random.uniform(0, 1, self.dim)  # Update quantum state
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness
                        # Adaptive adjustments based on success
                        self.F_max = min(1.0, self.F_max + self.alpha * (self.F_max - self.F_min))
                        self.CR_min = max(0.0, self.CR_min - self.alpha * (self.CR_max - self.CR_min))
        
        return best_individual, best_fitness