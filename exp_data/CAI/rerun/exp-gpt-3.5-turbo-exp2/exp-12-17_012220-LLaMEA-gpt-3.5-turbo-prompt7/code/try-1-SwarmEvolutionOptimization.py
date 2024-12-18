class SwarmEvolutionOptimization:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_rate=0.9, inertia_weight=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.inertia_weight = inertia_weight

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([func(ind) for ind in swarm])]
        best_fitness = func(best_position)

        for _ in range(self.budget):
            mutation_factor = self.mutation_factor * np.exp(-0.1 * _)  # Dynamic adjustment of mutation factor
            for i in range(self.swarm_size):
                r1, r2 = np.random.uniform(0, 1, size=2)
                velocities[i] = self.inertia_weight * velocities[i] + mutation_factor * (best_position - swarm[i]) + \
                                self.crossover_rate * (swarm[np.argsort([func(ind) for ind in swarm])[0]] - swarm[i])
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], -5.0, 5.0)

                fitness = func(swarm[i])
                if fitness < best_fitness:
                    best_position = swarm[i]
                    best_fitness = fitness

        return best_position