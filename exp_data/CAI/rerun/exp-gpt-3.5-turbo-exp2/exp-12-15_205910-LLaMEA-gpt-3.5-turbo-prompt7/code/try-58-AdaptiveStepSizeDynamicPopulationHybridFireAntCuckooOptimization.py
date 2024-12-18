class AdaptiveStepSizeDynamicPopulationHybridFireAntCuckooOptimization(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        population_size = 10
        successful_steps = 0
        total_steps = 0
        
        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:  
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
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
                successful_steps += 1
            else:
                self.step_size *= 1.02  # Increase step size for unsuccessful steps to explore more space
                
            total_steps += 1
            if total_steps % 10 == 0:  # Adaptive step size control every 10 steps
                success_rate = successful_steps / 10
                if success_rate < 0.2:
                    self.step_size *= 0.9
                elif success_rate > 0.8:
                    self.step_size *= 1.1
                successful_steps = 0
        
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
        
        return best_solution