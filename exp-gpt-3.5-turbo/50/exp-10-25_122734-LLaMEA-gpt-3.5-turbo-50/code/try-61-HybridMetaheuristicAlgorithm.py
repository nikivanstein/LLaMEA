import numpy as np

class HybridMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size

    def differential_evolution(self, func, bounds):
        def constrain(x):
            return np.clip(x, bounds[0], bounds[1])

        pop = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = constrain(a + 0.5 * (b - c))
                crossover = np.random.rand(self.dim) < 0.9
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

        return pop[np.argmin(fitness)]

    def particle_swarm_optimization(self, func, bounds):
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

        for _ in range(self.max_iter):
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

    def __call__(self, func):
        bounds = (-5.0, 5.0)
        return self.differential_evolution(func, bounds) + self.particle_swarm_optimization(func, bounds)