import numpy as np

class EnhancedNovelMetaheuristicAlgorithm(NovelMetaheuristicAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return self.adaptive_differential_evolution(func, bounds) + self.adaptive_particle_swarm_optimization(func, bounds)

    def adaptive_differential_evolution(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        mutation_factor = 0.5

        for _ in range(self.max_iter):
            mutation_factor = self.adapt_parameter(mutation_factor)
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = constrain(a + mutation_factor * (b - c))
                crossover = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

        return pop[np.argmin(fitness)]

    def adaptive_particle_swarm_optimization(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        inertia = 0.5
        c1 = 1.5
        c2 = 1.5
        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = pop.copy()
        pbest_fit = np.array([func(ind) for ind in pop])
        gbest = pbest[np.argmin(pbest_fit)]
        inertia_factor = 0.5

        for _ in range(self.max_iter):
            inertia_factor = self.adapt_parameter(inertia_factor)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = inertia_factor * velocity[i] + c1 * r1 * (pbest[i] - pop[i]) + c2 * r2 * (gbest - pop[i])
                pop[i] = constrain(pop[i] + velocity[i])
                fit = func(pop[i])
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i]
                    pbest_fit[i] = fit
                    if fit < func(gbest):
                        gbest = pop[i]

        return gbest

    def adapt_parameter(self, param):
        if np.random.rand() < 0.5:
            return max(0.1, param - 0.1)  # Decrease parameter
        else:
            return min(0.9, param + 0.1)  # Increase parameter