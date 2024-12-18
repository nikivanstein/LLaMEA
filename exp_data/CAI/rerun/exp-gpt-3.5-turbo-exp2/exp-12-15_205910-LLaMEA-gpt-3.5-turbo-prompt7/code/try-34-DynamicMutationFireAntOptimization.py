class DynamicMutationFireAntOptimization(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_rate = 0.1
        
        for _ in range(self.budget):
            if np.random.rand() < mutation_rate:  # Introduce dynamic mutation based on fitness improvement rate
                current_fitness = func(best_solution)
                mutated_solution = np.clip(best_solution + np.random.normal(0, mutation_rate, self.dim), self.lower_bound, self.upper_bound)
                new_fitness = func(mutated_solution)
                mutation_rate *= 1.01 if new_fitness < current_fitness else 0.99
                best_solution = mutated_solution
                best_fitness = new_fitness
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.99
                if _ % 100 == 0:
                    self.step_size *= 1.01
        
        return best_solution