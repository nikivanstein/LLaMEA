class DynamicMutationHybridFireAntCuckooOptimizationImproved(DynamicMutationHybridFireAntCuckooOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 10

    def __call__(self, func):
        best_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        best_fitness = func(best_solution)
        mutation_prob = 0.2
        
        for _ in range(self.budget):
            if np.random.rand() < mutation_prob:  # Dynamic mutation probability
                best_solution = np.clip(best_solution + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
                best_fitness = func(best_solution)
                mutation_prob *= 0.99  # Adjust mutation probability based on fitness improvement rate
            
            steps = self.step_size * np.random.uniform(-1, 1, (self.dim, self.dim))
            new_solutions = np.clip(best_solution + steps, self.lower_bound, self.upper_bound)
            new_fitnesses = np.array([func(sol) for sol in new_solutions])
            
            min_idx = np.argmin(new_fitnesses)
            if new_fitnesses[min_idx] < best_fitness:
                best_solution = new_solutions[min_idx]
                best_fitness = new_fitnesses[min_idx]
                self.step_size *= 0.99
                if _ % 100 == 0:  # Dynamic step size adaptation
                    self.step_size *= 1.01
            
            # Cuckoo search exploration
            cuckoo_solution = np.clip(best_solution + np.random.uniform(-0.1, 0.1, self.dim), self.lower_bound, self.upper_bound)
            cuckoo_fitness = func(cuckoo_solution)
            if cuckoo_fitness < best_fitness:
                best_solution = cuckoo_solution
                best_fitness = cuckoo_fitness
            
            # Dynamic population size adaptation
            if best_fitness < 0.1 and self.population_size > 5:
                self.population_size -= 1
            elif best_fitness > 0.5 and self.population_size < 15:
                self.population_size += 1
            
        return best_solution