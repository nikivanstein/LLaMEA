import numpy as np

class DynamicNeighborhoodSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        mutation_rates = np.full(pop_size, 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                neighbors = [ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim) for _ in range(5)]
                
                best_neighbor_fitness = min([func(nb) for nb in neighbors])
                best_neighbor_idx = np.argmin([func(nb) for nb in neighbors])
                
                if best_neighbor_fitness < fitness_values[idx]:
                    population[idx] = neighbors[best_neighbor_idx]
                    fitness_values[idx] = best_neighbor_fitness
                    scaling_factors[idx] *= 1.1
                    if np.random.uniform(0, 1) < 0.2:
                        mutation_rates[idx] *= 1.2
                    else:
                        mutation_rates[idx] *= 0.9
                else:
                    scaling_factors[idx] *= 0.9
                    mutation_rates[idx] *= 0.8
                
                if best_neighbor_fitness < best_fitness:
                    best_solution = neighbors[best_neighbor_idx]
                    best_fitness = best_neighbor_fitness
        
        return best_solution