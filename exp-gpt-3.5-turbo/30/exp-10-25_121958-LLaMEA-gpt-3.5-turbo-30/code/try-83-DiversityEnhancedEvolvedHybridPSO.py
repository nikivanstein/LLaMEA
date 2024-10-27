import numpy as np

class DiversityEnhancedEvolvedHybridPSO(EvolvedHybridPSO):
    def __init__(self, budget, dim, n_particles=30, max_local_iter=10, mutation_rate=0.1, adaptive_rate=0.3, diversity_rate=0.5):
        super().__init__(budget, dim, n_particles, max_local_iter, mutation_rate, adaptive_rate)
        self.diversity_rate = diversity_rate

    def __call__(self, func):
        def update_particle(particle, func):
            current_fitness = func(particle)
            new_particle = particle + np.random.uniform(-1, 1, size=self.dim) * (best_global_position - particle)
            new_particle = np.clip(new_particle, -5.0, 5.0)

            if np.random.rand() < self.mutation_rate:
                new_particle += np.random.normal(0, 0.5, size=self.dim)

            if np.random.rand() < self.diversity_rate:
                new_particle = adaptive_local_search(new_particle)

            new_fitness = func(new_particle)
            if new_fitness < current_fitness:
                return new_particle
            else:
                return particle

        swarm = np.random.uniform(-5.0, 5.0, size=(self.n_particles, self.dim))
        swarm_fitness = np.array([func(p) for p in swarm])
        best_idx = np.argmin(swarm_fitness)
        best_global_position = swarm[best_idx]

        for _ in range(self.budget // self.n_particles):
            for i in range(self.n_particles):
                swarm[i] = update_particle(swarm[i], func)
                if func(swarm[i]) < func(best_global_position):
                    best_global_position = swarm[i]

        return best_global_position