class ImprovedSwarmEvolutionOptimization:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_rate=0.9, initial_inertia_weight=0.5, elitism_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.inertia_weight = initial_inertia_weight
        self.elitism_rate = elitism_rate

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([func(ind) for ind in swarm])]
        best_fitness = func(best_position)

        for t in range(0, self.budget, self.swarm_size):
            mutation_factor = self.mutation_factor * np.exp(-0.1 * t)
            adaptive_inertia_weight = self.inertia_weight * np.exp(-0.001 * t)  # Adaptive inertia weight
            for i in range(self.swarm_size):
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocities[i] = adaptive_inertia_weight * velocities[i] + mutation_factor * (best_position - swarm[i]) + \
                                self.crossover_rate * (swarm[np.argsort([func(ind) for ind in swarm])[0]] - swarm[i])
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

            fitness_values = [func(ind) for ind in swarm]
            for i in range(self.swarm_size):
                if fitness_values[i] < best_fitness:
                    best_position = swarm[i]
                    best_fitness = fitness_values[i]
            
            # Introducing Elitism
            num_elites = max(1, int(self.elitism_rate * self.swarm_size))
            elite_indices = np.argsort(fitness_values)[:num_elites]
            for elite_idx in elite_indices:
                if fitness_values[elite_idx] < best_fitness:
                    best_position = swarm[elite_idx]
                    best_fitness = fitness_values[elite_idx]

        return best_position