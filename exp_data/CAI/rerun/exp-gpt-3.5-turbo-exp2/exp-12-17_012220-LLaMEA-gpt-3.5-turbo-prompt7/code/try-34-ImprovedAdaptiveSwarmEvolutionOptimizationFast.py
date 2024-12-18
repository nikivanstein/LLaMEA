import concurrent.futures

class ImprovedAdaptiveSwarmEvolutionOptimizationFast(ImprovedSwarmEvolutionOptimization):
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_rate=0.9, initial_inertia_weight=0.5, local_search_prob=0.3):
        super().__init__(budget, dim, swarm_size, mutation_factor, crossover_rate, initial_inertia_weight, local_search_prob)
    
    def __call__(self, func):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(func, np.random.uniform(-5.0, 5.0, size=self.dim)): i for i in range(self.swarm_size)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                fitness = future.result()
                if fitness < best_fitness:
                    best_position = swarm[idx]
                    best_fitness = fitness
        return best_position