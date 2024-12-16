from concurrent.futures import ThreadPoolExecutor

class ParallelPopulationSizeHybridFireAntCuckooOptimizationImproved(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        population_size = 10
        
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                if np.random.rand() < mutation_prob:  
                    mutation_scale = 0.1
                    rand_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    mutated_solution = best_solution + mutation_scale * (rand_solution - best_solution)
                    best_fitness = func(mutated_solution)
                    mutation_prob *= 0.99  
                    self.step_size *= 0.99  # Adjust step size based on fitness improvement rate
                    
                steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
                new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
                
                def evaluate(sol):
                    return func(sol)
                
                new_fitnesses = np.array(list(executor.map(evaluate, new_solutions)))
                
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