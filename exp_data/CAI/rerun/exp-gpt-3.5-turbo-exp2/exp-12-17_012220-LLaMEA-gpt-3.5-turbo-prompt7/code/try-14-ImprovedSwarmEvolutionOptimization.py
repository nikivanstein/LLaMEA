from concurrent.futures import ThreadPoolExecutor

class ImprovedSwarmEvolutionOptimization:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_rate=0.9, initial_inertia_weight=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.inertia_weight = initial_inertia_weight

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_position = swarm[np.argmin([func(ind) for ind in swarm])]
        best_fitness = func(best_position)

        with ThreadPoolExecutor() as executor:
            for t in range(0, self.budget, self.swarm_size):
                mutation_factor = self.mutation_factor * np.exp(-0.1 * t)
                adaptive_inertia_weight = self.inertia_weight * np.exp(-0.001 * t)  # Adaptive inertia weight
                futures = []
                for i in range(self.swarm_size):
                    r1, r2 = np.random.uniform(0, 1, size=2)
                    futures.append(executor.submit(self._update_swarm, func, swarm, velocities, best_position, mutation_factor, adaptive_inertia_weight, i))

                results = [future.result() for future in futures]
                swarm, velocities, fitness_values = zip(*results)

                best_index = np.argmin(fitness_values)
                if fitness_values[best_index] < best_fitness:
                    best_position = swarm[best_index]
                    best_fitness = fitness_values[best_index]

        return best_position

    def _update_swarm(self, func, swarm, velocities, best_position, mutation_factor, adaptive_inertia_weight, i):
        velocities[i] = adaptive_inertia_weight * velocities[i] + mutation_factor * (best_position - swarm[i]) + \
                        self.crossover_rate * (swarm[np.argsort([func(ind) for ind in swarm])[0]] - swarm[i])
        new_position = swarm[i] + velocities[i]
        new_position = np.clip(new_position, -5.0, 5.0)
        fitness_value = func(new_position)
        return new_position, velocities[i], fitness_value