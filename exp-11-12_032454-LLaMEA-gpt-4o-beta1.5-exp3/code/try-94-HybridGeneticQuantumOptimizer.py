import numpy as np

class HybridGeneticQuantumOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = max(5, int(budget / (10 * dim)))  # heuristic for population size
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.quantum_probability = 0.5
        
    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        quantum_state = np.random.uniform(0, 1, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        num_evaluations = self.population_size
        
        global_best_idx = np.argmin(fitness)
        global_best_position = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]
        
        while num_evaluations < self.budget:
            new_population = []
            
            for _ in range(self.population_size // 2):
                if num_evaluations >= self.budget:
                    break
                
                # Select parents
                parents = np.random.choice(self.population_size, 2, replace=False)
                parent1, parent2 = population[parents]
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim-1)
                    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                for child in [child1, child2]:
                    if np.random.rand() < self.mutation_rate:
                        mutation_vector = np.random.normal(0, 1, self.dim)
                        child += mutation_vector
                        child = np.clip(child, self.lb, self.ub)
                    
                    # Quantum effect
                    quantum_effect = np.random.rand(self.dim) < quantum_state[parents[0]]
                    child = np.where(quantum_effect, child, global_best_position)
                    
                    # Evaluate child
                    child_fitness = func(child)
                    num_evaluations += 1
                    
                    # Update global best
                    if child_fitness < global_best_fitness:
                        global_best_position = child
                        global_best_fitness = child_fitness
                    
                    new_population.append(child)
                    if num_evaluations >= self.budget:
                        break
            
            if num_evaluations >= self.budget:
                break
            
            # Select next generation
            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])
            num_evaluations += len(population)
            
            # Update quantum states
            quantum_state = np.random.uniform(0, 1, (self.population_size, self.dim))
        
        return global_best_position, global_best_fitness