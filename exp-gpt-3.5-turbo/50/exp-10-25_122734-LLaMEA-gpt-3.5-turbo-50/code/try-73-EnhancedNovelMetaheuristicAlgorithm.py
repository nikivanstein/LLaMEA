import numpy as np

class EnhancedNovelMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def particle_swarm_optimization(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        inertia_min = 0.4
        inertia_max = 0.9
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pop])
        gbest = pbest[np.argmin(pbest_fit)]
        
        for _ in range(self.max_iter):
            inertia = inertia_min + (inertia_max - inertia_min) * (_ / self.max_iter)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = inertia * velocity[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                pop[i] = constrain(pop[i] + velocity[i])
                fit = func(pop[i])
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i]
                    pbest_fit[i] = fit
                    if fit < func(gbest):
                        gbest = pop[i]

        return gbest