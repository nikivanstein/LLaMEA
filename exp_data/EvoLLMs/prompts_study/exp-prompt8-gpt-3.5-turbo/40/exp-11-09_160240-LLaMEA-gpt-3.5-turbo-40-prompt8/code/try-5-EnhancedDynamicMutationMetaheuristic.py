import numpy as np

class EnhancedDynamicMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        mutation_scaling_factor = 0.5
        mutation_stage = 1
        mutation_rate = 0.5
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            diversity = np.std(population)
            if diversity < 1e-6:
                mutation_stage = 2
            elif diversity > 0.1:
                mutation_stage = 1
            
            for idx, ind in enumerate(population):
                if mutation_stage == 1:
                    mutated_solution = ind + mutation_rate * np.random.normal(0, 1, self.dim)
                else:
                    idxs = np.random.choice(range(pop_size), 2, replace=False)
                    donor = population[idxs[0]] + 0.5 * (population[idxs[1]] - ind)
                    mutated_solution = ind + mutation_rate * (donor - ind)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
        
        return best_solution