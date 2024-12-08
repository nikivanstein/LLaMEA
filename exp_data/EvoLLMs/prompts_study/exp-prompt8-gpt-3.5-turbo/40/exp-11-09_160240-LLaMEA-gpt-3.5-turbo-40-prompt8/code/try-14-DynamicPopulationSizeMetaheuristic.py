import numpy as np

class DynamicPopulationSizeMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        scaling_factors = np.full(10, 0.5)
        mutation_rates = np.full(10, 0.5)
        pop_size = 10
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                mutated_solution = ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                    scaling_factors[idx] *= 1.1
                    if np.random.uniform(0, 1) < 0.2:  # Perturb a percentage of population
                        mutation_rates[idx] *= 1.2
                    else:
                        mutation_rates[idx] *= 0.9
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
            
            if np.random.uniform(0, 1) < 0.1:  # Dynamic population size adjustment
                new_pop_size = max(2, min(20, int(pop_size * np.random.normal(1, 0.5))))
                population = population[:new_pop_size]
                fitness_values = fitness_values[:new_pop_size]
                scaling_factors = np.concatenate((scaling_factors, np.full(new_pop_size - pop_size, 0.5)))
                mutation_rates = np.concatenate((mutation_rates, np.full(new_pop_size - pop_size, 0.5)))
                pop_size = new_pop_size
        
        return best_solution