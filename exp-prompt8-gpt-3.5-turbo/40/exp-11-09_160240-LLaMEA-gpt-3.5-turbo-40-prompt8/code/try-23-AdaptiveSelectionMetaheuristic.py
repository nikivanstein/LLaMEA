import numpy as np

class AdaptiveSelectionMetaheuristic:
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
            
            for idx in range(pop_size):
                selected_idx = np.random.choice(range(pop_size), p=[1 - (fitness_values[i] - min(fitness_values))/(max(fitness_values) - min(fitness_values)) for i in range(pop_size)])
                mutated_solution = population[selected_idx] + scaling_factors[selected_idx] * np.random.normal(0, 1, self.dim)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[selected_idx]:
                    population[selected_idx] = mutated_solution
                    fitness_values[selected_idx] = fitness
                    scaling_factors[selected_idx] *= 1.1
                    if np.random.uniform(0, 1) < 0.2:
                        mutation_rates[selected_idx] *= 1.2
                    else:
                        mutation_rates[selected_idx] *= 0.9
                else:
                    scaling_factors[selected_idx] *= 0.9
                    mutation_rates[selected_idx] *= 0.8
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
        
        return best_solution