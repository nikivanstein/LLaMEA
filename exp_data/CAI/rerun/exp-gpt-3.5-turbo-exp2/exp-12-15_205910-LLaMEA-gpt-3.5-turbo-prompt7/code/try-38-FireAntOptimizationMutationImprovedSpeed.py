class FireAntOptimizationMutationImprovedSpeed(FireAntOptimizationMutationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        population_size = 10
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:  # Introduce random mutation with 20% probability
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
            
            steps = self.step_size * np.random.uniform(-1, 1, (population_size, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.99
                if _ % 100 == 0:  # Dynamic step size adaptation
                    self.step_size *= 1.01
                    population_size += 5  # Increase population size for faster convergence
        
        return best_solution