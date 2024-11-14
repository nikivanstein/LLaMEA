import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (8 * dim)))  # heuristic for population size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]
        
        while num_evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if num_evaluations >= self.budget:
                    break
                
                # Select three random individuals for mutation
                indices = np.random.choice([idx for idx in range(self.population_size) if idx != i], 3, replace=False)
                a, b, c = population[indices]
                
                # Quantum-inspired mutation
                mutant_vector = a + self.mutation_factor * (b - c)
                quantum_superposition = np.random.choice([mutant_vector, population[i]], p=[0.5, 0.5])
                
                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, quantum_superposition, population[i])
                trial_vector = np.clip(trial_vector, self.lb, self.ub)
                
                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                num_evaluations += 1
                
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial_vector
                        best_fitness = trial_fitness
            
            population = new_population
        
        return best_solution, best_fitness