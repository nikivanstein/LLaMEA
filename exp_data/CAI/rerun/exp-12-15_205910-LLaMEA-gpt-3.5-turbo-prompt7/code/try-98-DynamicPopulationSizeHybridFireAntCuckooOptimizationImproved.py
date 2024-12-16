class DynamicPopulationSizeHybridFireAntCuckooOptimizationImproved(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        
        for _ in range(self.budget):
            population_size = np.clip(int(10 + np.random.randn()), 1, 20)  # Dynamic population size adjustment
            if np.random.rand() < mutation_prob:  
                mutation_scale = 0.1 + 0.05 * np.random.randn()  # Dynamic mutation scale adjustment
                rand_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                mutated_solution = best_solution + mutation_scale * (rand_solution - best_solution)
                best_fitness = func(mutated_solution)
                mutation_prob *= 0.99  
                self.step_size *= 0.99  # Adjust step size based on fitness improvement rate
                
            steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.98  # Dynamic step size adjustment for faster convergence
                
            global_best_solution = new_solutions[np.argmin(new_fitnesses)]
            combined_solution = best_solution + self.step_size * (global_best_solution - best_solution) * np.random.uniform(0.5, 1.0, self.dim)
            combined_solution = np.clip(combined_solution, self.lower_bound, self.upper_bound)
            combined_fitness = func(combined_solution)
            
            if combined_fitness < best_fitness:
                best_solution = combined_solution
                best_fitness = combined_fitness
        
        return best_solution