class EnhancedAdaptiveStepSizeMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        step_sizes = np.full(pop_size, 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                exploration_factor = np.exp(-0.1 * fitness_values[idx])  # Adjust exploration based on fitness
                step_size = step_sizes[idx] * exploration_factor
                
                mutated_solution = ind + step_size * np.random.normal(0, 1, self.dim)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                    step_sizes[idx] *= 1.1
                else:
                    step_sizes[idx] *= 0.9  # Decrease step size for non-improving solutions
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
        
        return best_solution