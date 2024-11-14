class DynamicEnsembleMutationMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        mutation_operators = [self.mutation_strategy_1, self.mutation_strategy_2, self.mutation_strategy_3]
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            for idx, ind in enumerate(population):
                selected_operator = np.random.choice(mutation_operators)
                mutated_solution = selected_operator(ind)
                
                fitness = func(mutated_solution)
                if fitness < fitness_values[idx]:
                    population[idx] = mutated_solution
                    fitness_values[idx] = fitness
                
                    if np.random.uniform(0, 1) < 0.2:
                        self.adapt_mutation_parameters()
                
                if fitness < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = fitness
        
        return best_solution
    
    def mutation_strategy_1(self, solution):
        return solution + 0.1 * np.random.normal(0, 1, self.dim)
    
    def mutation_strategy_2(self, solution):
        return solution + 0.2 * np.random.normal(0, 1, self.dim)
    
    def mutation_strategy_3(self, solution):
        return solution + 0.3 * np.random.normal(0, 1, self.dim)
    
    def adapt_mutation_parameters(self):
        # Custom logic to adapt mutation parameters dynamically
        pass