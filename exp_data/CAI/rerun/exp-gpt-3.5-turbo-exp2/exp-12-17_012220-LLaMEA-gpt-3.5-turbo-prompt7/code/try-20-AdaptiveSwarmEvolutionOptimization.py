class AdaptiveSwarmEvolutionOptimization(ImprovedSwarmEvolutionOptimization):
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_rate=0.9, initial_inertia_weight=0.5, local_search_prob=0.3):
        super().__init__(budget, dim, swarm_size, mutation_factor, crossover_rate, initial_inertia_weight, local_search_prob)
    
    def __call__(self, func):
        for t in range(0, self.budget, self.swarm_size):
            self.swarm_size = min(50, int(30 + 0.2 * t))  # Dynamic swarm size adjustment
            swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
            velocities = np.zeros((self.swarm_size, self.dim))
            best_position = swarm[np.argmin([func(ind) for ind in swarm])]
            best_fitness = func(best_position)
            mutation_factor = self.mutation_factor * np.exp(-0.1 * t)
            adaptive_inertia_weight = self.inertia_weight * np.exp(-0.001 * t)  # Adaptive inertia weight
            for i in range(self.swarm_size):
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocities[i] = adaptive_inertia_weight * velocities[i] + mutation_factor * (best_position - swarm[i]) + \
                                self.crossover_rate * (swarm[np.argsort([func(ind) for ind in swarm])[0]] - swarm[i])
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)
                if np.random.rand() < self.local_search_prob:
                    candidate_position = swarm[i] + np.random.normal(0, 0.1, size=self.dim)
                    candidate_position = np.clip(candidate_position, -5.0, 5.0)
                    if func(candidate_position) < func(swarm[i]):
                        swarm[i] = candidate_position
            fitness_values = [func(ind) for ind in swarm]
            for i in range(self.swarm_size):
                if fitness_values[i] < best_fitness:
                    best_position = swarm[i]
                    best_fitness = fitness_values[i]
        return best_position