class FireAntOptimizationMutationImproved(FireAntOptimizationImproved):
    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            if np.random.rand() < 0.2:
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            
            def evaluate_fitness(sol):
                return func(sol)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                new_fitnesses = np.array(list(executor.map(evaluate_fitness, new_solutions)))
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.99
                if _ % 100 == 0:
                    self.step_size *= 1.01
        
        return best_solution