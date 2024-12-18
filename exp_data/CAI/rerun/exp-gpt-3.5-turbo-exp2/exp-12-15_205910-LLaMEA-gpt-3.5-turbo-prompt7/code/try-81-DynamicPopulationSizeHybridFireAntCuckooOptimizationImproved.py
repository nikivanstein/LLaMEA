class DynamicPopulationSizeHybridFireAntCuckooOptimizationImproved(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.4  # Increased mutation probability
        population_size = 10
        
        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:  
                best_solution = np.clip(best_solution + np.random.normal(0, 0.2, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
                mutation_prob *= 0.98  # Faster decrease in mutation probability
                self.step_size *= 0.98  # Adjust step size based on fitness improvement rate more aggressively
                
            steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.96  # More aggressive dynamic step size adjustment for faster convergence
                
            global_best_solution = new_solutions[np.argmin(new_fitnesses)]
            combined_solution = best_solution + self.step_size * (global_best_solution - best_solution) * np.random.uniform(0.6, 1.2, self.dim)  # Increased mutation range
            combined_solution = np.clip(combined_solution, self.lower_bound, self.upper_bound)
            combined_fitness = func(combined_solution)
            
            if combined_fitness < best_fitness:
                best_solution = combined_solution
                best_fitness = combined_fitness
        
        return best_solution