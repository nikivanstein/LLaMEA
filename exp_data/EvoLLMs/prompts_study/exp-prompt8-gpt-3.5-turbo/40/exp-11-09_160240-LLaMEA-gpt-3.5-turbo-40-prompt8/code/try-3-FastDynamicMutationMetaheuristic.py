import numpy as np

class FastDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        mutation_scaling_factor = 0.5
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            avg_fitness_diff = np.mean([abs(f - best_fitness) for f in fitness_values])
            mutation_rate = mutation_scaling_factor / (1 + avg_fitness_diff)
            
            for idx, ind in enumerate(population):
                mutated_solution = ind + mutation_rate * np.random.normal(0, 1, self.dim)
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                
                    if fitness < best_fitness:
                        best_solution = mutated_solution
                        best_fitness = fitness
        
        return best_solution