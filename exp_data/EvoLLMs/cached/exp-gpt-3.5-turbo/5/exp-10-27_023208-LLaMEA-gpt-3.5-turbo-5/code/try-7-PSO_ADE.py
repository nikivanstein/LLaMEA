import numpy as np

class PSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.p_best_rate = 0.3
        self.mutation_factor = 0.5

    def _mutation(self, population, target_index):
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return mutant

    def _update_velocity(self, swarm, p_best, g_best):
        r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
        return self.w * swarm + self.c1 * r1 * (p_best - swarm) + self.c2 * r2 * (g_best - swarm)

    def _optimize_func(self, func, swarm):
        p_best = swarm[np.argsort([func(p) for p in swarm])[:int(self.p_best_rate * self.swarm_size)]]
        g_best = p_best[0]
        for _ in range(self.budget):
            velocities = self._update_velocity(swarm, p_best, g_best)
            swarm = swarm + velocities
            for idx, particle in enumerate(swarm):
                mutant = self._mutation(swarm, idx)
                trial = np.where(np.random.rand(self.dim) < 0.5, mutant, particle)
                if func(trial) < func(particle):
                    swarm[idx] = trial
            p_best = swarm[np.argsort([func(p) for p in swarm])[:int(self.p_best_rate * self.swarm_size)]]
            g_best = p_best[0]
        return g_best

    def __call__(self, func):
        swarm = np.random.uniform(-5, 5, (self.swarm_size, self.dim))
        return self._optimize_func(func, swarm)