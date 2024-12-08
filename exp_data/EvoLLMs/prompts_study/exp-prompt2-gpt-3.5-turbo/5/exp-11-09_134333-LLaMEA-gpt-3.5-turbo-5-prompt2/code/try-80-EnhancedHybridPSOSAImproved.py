class EnhancedHybridPSOSAImproved(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def pso_search(best_solution, inertia_weight):
            mutation_rate = np.random.uniform(0.1, 1.0) ** 2
            new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim) * inertia_weight
            return new_solution
        
        def sa_search(best_solution):
            mutation_rate = np.random.uniform(0.1, 1.0) ** 2
            new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution
        
        def opposition_based_search(solution):
            opposite_solution = -solution
            return solution if func(solution) < func(opposite_solution) else opposite_solution
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        inertia_weight = 0.5  # Initial inertia weight
        
        for _ in range(self.budget):
            selected_search = pso_search if np.random.rand() < 0.4 else (sa_search if np.random.rand() < 0.8 else opposition_based_search)
            new_solution = selected_search(best_solution, inertia_weight)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                inertia_weight = max(0.4, inertia_weight * 0.99)  # Update inertia weight dynamically
        
        return best_solution