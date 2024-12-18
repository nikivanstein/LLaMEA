class ImprovedDynamicPopulationSizeHybridFireAntCuckooOptimization(DynamicPopulationSizeHybridFireAntCuckooOptimization):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        population_size = 10
        
        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:  
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
                mutation_prob *= 0.99  
                self.step_size *= 0.99  # Adjust step size based on fitness improvement rate
                
            # Enhanced mutation mechanism
            steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim)) + 0.1 * np.random.uniform(-1, 1, (1, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.98  # Dynamic step size adjustment for faster convergence
                
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
        
        return best_solution