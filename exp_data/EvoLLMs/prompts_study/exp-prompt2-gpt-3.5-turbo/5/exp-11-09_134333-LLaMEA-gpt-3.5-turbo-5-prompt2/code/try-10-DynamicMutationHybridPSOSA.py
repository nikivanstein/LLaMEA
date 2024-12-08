class DynamicMutationHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def pso_search(best_solution, mutation_rate):
            new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim)
            return new_solution
        
        def sa_search(best_solution, mutation_rate):
            new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        mutation_rate = np.random.uniform(0.1, 1.0)
        for _ in range(self.budget):
            new_solution = pso_search(best_solution, mutation_rate) if np.random.rand() < 0.5 else sa_search(best_solution, mutation_rate)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                mutation_rate = mutation_rate * 1.1  # Increase mutation rate when fitness improves
        
        return best_solution