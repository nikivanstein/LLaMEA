class ImprovedHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.w = 0.9

    def pso_sa_optimization():
        swarm_best_fitness = np.inf
        best_position = None
        T = 1.0
        for _ in range(self.budget):
            swarm_position = initialize_population()
            for _ in range(self.dim):
                new_position = mutate(swarm_position, T)
                new_fitness = objective_function(new_position)
                if new_fitness < swarm_best_fitness:
                    swarm_best_fitness = new_fitness
                    best_position = new_position
            self.w = max(0.4, self.w - 0.01)  # Dynamic inertia weight update
            T = self.w
        return best_position