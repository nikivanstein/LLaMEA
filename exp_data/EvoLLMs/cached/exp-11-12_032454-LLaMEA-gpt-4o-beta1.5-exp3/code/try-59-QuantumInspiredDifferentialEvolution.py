import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.F = 0.5  # scaling factor for mutation
        self.CR = 0.9  # crossover probability
        
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
                
                # Mutation using Quantum-inspired differential
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutated_vector = population[a] + self.F * (population[b] - population[c])
                mutated_vector = np.clip(mutated_vector, self.lb, self.ub)
                
                # Quantum crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                offspring = np.where(crossover_mask, mutated_vector, population[i])
                
                # Superposition-based enhancement
                quantum_selection = np.random.rand(self.dim) < quantum_population[i]
                offspring = np.where(quantum_selection, offspring, best_individual)
                
                # Evaluate offspring
                offspring_fitness = func(offspring)
                num_evaluations += 1
                
                # Selection
                if offspring_fitness < fitness[i]:
                    population[i] = offspring
                    fitness[i] = offspring_fitness
                    quantum_population[i] = np.random.uniform(0, 1, self.dim)  # Update quantum state
                    if offspring_fitness < best_fitness:
                        best_individual = offspring
                        best_fitness = offspring_fitness
        
        return best_individual, best_fitness