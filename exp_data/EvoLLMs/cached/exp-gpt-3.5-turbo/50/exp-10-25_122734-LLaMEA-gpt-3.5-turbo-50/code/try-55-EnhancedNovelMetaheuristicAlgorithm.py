import numpy as np

class EnhancedNovelMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def greedy_selection(self, func, bounds, elite_ratio=0.1):
        elite_size = int(self.pop_size * elite_ratio)
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        for _ in range(self.max_iter):
            elite_idxs = np.argsort(fitness)[:elite_size]
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + 0.5 * (b - c), bounds[0], bounds[1])
                crossover = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

        return pop[elite_idxs]

    def __call__(self, func):
        bounds = (-5.0, 5.0)
        de_best = self.differential_evolution(func, bounds)
        pso_best = self.particle_swarm_optimization(func, bounds)
        elite_pop = self.greedy_selection(func, bounds)
        return (de_best + pso_best + elite_pop) / 3