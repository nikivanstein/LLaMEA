class FireAntOptimizationMutationImproved(FireAntOptimizationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        fitness_history = [best_fitness]
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:  # Introduce random mutation with 20% probability
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                fitness_history.append(best_fitness)
                if len(fitness_history) > 1:
                    fitness_rate = (fitness_history[-2] - fitness_history[-1]) / fitness_history[-2]
                    self.step_size *= (1 + 0.5 * fitness_rate)
        
        return best_solution