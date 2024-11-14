import numpy as np

class DynamicEnsembleMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        num_subpopulations = 5
        subpop_size = 10
        scaling_factors = np.full((num_subpopulations, subpop_size), 0.5)
        mutation_rates = np.full((num_subpopulations, subpop_size), 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // (num_subpopulations * subpop_size)):
            for i in range(num_subpopulations):
                population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(subpop_size)]
                fitness_values = [func(ind) for ind in population]

                best_idx = np.argmin(fitness_values)
                if fitness_values[best_idx] < best_fitness:
                    best_solution = population[best_idx]
                    best_fitness = fitness_values[best_idx]
                
                for idx, ind in enumerate(population):
                    mutated_solution = ind + scaling_factors[i][idx] * np.random.normal(0, 1, self.dim)
                    
                    fitness = func(mutated_solution)
                    if fitness < fitness_values[idx]:
                        population[idx] = mutated_solution
                        fitness_values[idx] = fitness
                        scaling_factors[i][idx] *= 1.1
                        if np.random.uniform(0, 1) < 0.2:
                            mutation_rates[i][idx] *= 1.2
                        else:
                            mutation_rates[i][idx] *= 0.9
                    else:
                        scaling_factors[i][idx] *= 0.9
                        mutation_rates[i][idx] *= 0.8
                
        return best_solution