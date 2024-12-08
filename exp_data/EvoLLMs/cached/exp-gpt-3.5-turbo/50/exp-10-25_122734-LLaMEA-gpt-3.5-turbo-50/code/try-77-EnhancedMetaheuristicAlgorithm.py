# import numpy as np

class EnhancedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def cuckoo_search(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        num_nests = self.pop_size
        pa = 0.25
        alpha = 0.1
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_nest_idx = np.argmin(fitness)
        best_nest = pop[best_nest_idx]

        for _ in range(self.max_iter):
            step_size = alpha * (bounds[1] - bounds[0])
            new_nests = pop.copy()

            for i in range(num_nests):
                step = step_size * np.random.randn(self.dim)
                new_nest = pop[i] + step
                new_nest = constrain(new_nest)

                if np.random.rand() < pa:
                    fit_new_nest = func(new_nest)
                    if fit_new_nest < fitness[i]:
                        new_nests[i] = new_nest
                        fitness[i] = fit_new_nest

                    if fit_new_nest < func(best_nest):
                        best_nest = new_nest

            pop = new_nests

        return best_nest

    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return 0.5 * self.differential_evolution(func, bounds) + 0.3 * self.particle_swarm_optimization(func, bounds) + 0.2 * self.cuckoo_search(func, bounds)