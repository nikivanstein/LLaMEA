class EnhancedHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def pso_search(best_solution):
            mutation_rate = np.random.uniform(0.1, 1.0) if np.random.rand() < 0.9 else np.random.uniform(0.5, 1.0)
            new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim)
            return new_solution
        
        def sa_search(best_solution):
            mutation_rate = np.random.uniform(0.1, 1.0) if np.random.rand() < 0.9 else np.random.uniform(0.5, 1.0)
            new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            new_solution = pso_search(best_solution) if np.random.rand() < 0.5 else sa_search(best_solution)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        
        return best_solution