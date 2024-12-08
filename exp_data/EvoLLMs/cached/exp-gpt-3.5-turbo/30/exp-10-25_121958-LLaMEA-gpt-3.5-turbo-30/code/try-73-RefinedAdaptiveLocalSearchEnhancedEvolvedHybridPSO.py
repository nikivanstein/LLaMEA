import numpy as np

class RefinedAdaptiveLocalSearchEnhancedEvolvedHybridPSO(AdaptiveLocalSearchEnhancedEvolvedHybridPSO):
    def __init__(self, budget, dim, n_particles=30, max_local_iter=10, mutation_rate=0.1, adaptive_rate=0.3, local_search_rate=0.5):
        super().__init__(budget, dim, n_particles, max_local_iter, mutation_rate, adaptive_rate, local_search_rate)

    def __call__(self, func):
        def adaptive_local_search(x):
            best_x = x.copy()
            best_fitness = func(x)
            for _ in range(self.max_local_iter):
                new_x = x + np.random.uniform(-self.adaptive_rate, self.adaptive_rate, size=self.dim)
                new_fitness = func(new_x)
                if new_fitness < best_fitness:
                    best_x = new_x
                    best_fitness = new_fitness
            return best_x

        swarm = np.random.uniform(-5.0, 5.0, size=(self.n_particles, self.dim))
        swarm_fitness = np.array([func(p) for p in swarm])
        best_idx = np.argmin(swarm_fitness)
        best_solution = swarm[best_idx]

        for _ in range(self.budget // self.n_particles):
            for i in range(self.n_particles):
                new_position = swarm[i] + np.random.uniform(-1, 1, size=self.dim) * (best_solution - swarm[i])
                new_position = np.clip(new_position, -5.0, 5.0)

                if np.random.rand() < self.mutation_rate:
                    new_position += np.random.normal(0, 0.5, size=self.dim)

                if np.random.rand() < self.local_search_rate:
                    new_position = adaptive_local_search(new_position)
                new_fitness = func(new_position)

                if new_fitness < swarm_fitness[i]:
                    swarm[i] = new_position
                    swarm_fitness[i] = new_fitness

                    if new_fitness < func(best_solution):
                        best_solution = new_position

        return best_solution