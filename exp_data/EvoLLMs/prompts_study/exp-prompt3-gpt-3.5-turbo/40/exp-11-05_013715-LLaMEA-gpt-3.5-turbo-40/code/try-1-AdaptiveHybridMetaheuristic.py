import numpy as np

class AdaptiveHybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iterations = budget // self.population_size

    def __call__(self, func):
        def de_mutate(x, pop, f=0.5):
            candidates = pop[np.random.choice(len(pop), 3, replace=False)]
            mutant = x + f * (candidates[0] - x) + f * (candidates[1] - candidates[2])
            return np.clip(mutant, -5.0, 5.0)

        def pso_update(x, pbest, gbest, w=0.7, c1=1.5, c2=1.5):
            nonlocal velocity
            inertia_weight = 0.4 + 0.5 * (self.max_iterations - _) / self.max_iterations
            velocity = inertia_weight * velocity \
                       + c1 * np.random.rand(self.dim) * (pbest - x) \
                       + c2 * np.random.rand(self.dim) * (gbest - x)
            x_new = np.clip(x + velocity, -5.0, 5.0)
            return x_new, velocity

        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        pbest = population.copy()
        gbest = population[np.argmin(fitness)]
        gbest_fitness = func(gbest)
        velocity = np.zeros(self.dim)

        for _ in range(self.max_iterations):
            for i in range(self.population_size):
                mutant = de_mutate(population[i], population)
                trial = pso_update(population[i], pbest[i], gbest)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                    if trial_fitness < gbest_fitness:
                        gbest = trial
                        gbest_fitness = trial_fitness

        return gbest