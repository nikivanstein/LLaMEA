class FireAntOptimizationMutationImprovedSpeed(FireAntOptimizationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_strength = 0.1
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:
                best_solution = np.clip(best_solution + np.random.normal(0, mutation_strength, self.dim), self.lower_bound, self.upper_bound)
                new_fitness = func(best_solution)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    mutation_strength *= 1.05  # Increase mutation strength if fitness improves
                else:
                    mutation_strength *= 0.95  # Decrease mutation strength if fitness worsens
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.99
        
        return best_solution