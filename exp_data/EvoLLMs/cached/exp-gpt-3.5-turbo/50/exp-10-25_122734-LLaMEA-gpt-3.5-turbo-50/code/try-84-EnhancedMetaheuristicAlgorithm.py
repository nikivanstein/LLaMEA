# import numpy as np

class EnhancedMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def firefly_algorithm(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        beta0 = 1.0
        beta_min = 0.2
        gamma = 0.01
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        attractiveness = beta0 * np.exp(-gamma * np.linalg.norm(pop[j] - pop[i])**2)
                        step = attractiveness * (pop[j] - pop[i])
                        pop[i] = constrain(pop[i] + step)
                        fitness[i] = func(pop[i])

        return pop[np.argmin(fitness)]
    
    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return self.differential_evolution(func, bounds) + self.particle_swarm_optimization(func, bounds) + self.firefly_algorithm(func, bounds)