import numpy as np

class DynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                mutation_rate = 1 - fitness_values[idx] / best_fitness
                mutated_solution = ind + scaling_factors[idx] * mutation_rate * np.random.normal(0, 1, self.dim)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                    scaling_factors[idx] *= 1.1
                else:
                    scaling_factors[idx] *= 0.9
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
        
        return best_solution