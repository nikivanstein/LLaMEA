import numpy as np

class DynamicChaosDEAPSO_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.max_iterations = budget // self.population_size
        self.w = 0.9
        self.w_min = 0.4
        self.w_max = 0.9
        self.mutation_prob = 0.2
        self.crossover_prob = 0.9
        self.inertia_weight = 0.5
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        def mutate(x, a, b, c, f):
            return np.clip(a + f * (b - c), -5.0, 5.0)

        def explore_mutate(x):
            return np.clip(x + np.random.normal(0, 1, x.shape), -5.0, 5.0)

        def self_adaptive_mutate(x, f):
            return np.clip(x + f * np.random.normal(0, 1, x.shape), -5.0, 5.0)

        def chaotic_search(x, best, chaos_param):
            new_x = x + chaos_param * np.random.uniform(-5.0, 5.0, x.shape)
            new_x = np.clip(new_x, -5.0, 5.0)
            if func(new_x) < func(x):
                return new_x
            else:
                return x

        def local_search(x, best, radius=0.1):
            x_new = np.clip(x + radius * np.random.normal(0, 1, x.shape), -5.0, 5.0)
            if func(x_new) < func(x):
                return x_new
            else:
                return x

        def differential_evolution(population, fitness, best, f, cr, chaos_param):
            new_population = np.copy(population)
            for i in range(self.population_size):
                a, b, c = population[np.random.choice(range(self.population_size), 3, replace=False)]
                x_new = mutate(population[i], a, b, c, f)
                if np.all(x_new == population[i]) or np.random.rand() < cr:
                    x_new = a + f * (b - c)
                fitness_new = func(x_new)
                if fitness_new < fitness[i]:
                    new_population[i] = x_new
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
                if np.random.rand() < self.mutation_prob:
                    new_population[i] = self_adaptive_mutate(new_population[i], f)
                new_population[i] = chaotic_search(new_population[i], best, chaos_param)
                new_population[i] = local_search(new_population[i], best)  # Integrate local search
            return new_population, fitness, best

        def particle_swarm_optimization(population, fitness, best, velocity, pbest, gbest):
            for i in range(self.population_size):
                r1, r2 = np.random.uniform(0, 1, 2)
                velocity[i] = self.inertia_weight*velocity[i] + self.c1*r1*(pbest[i]-population[i]) + self.c2*r2*(gbest-population[i])
                population[i] = np.clip(population[i] + velocity[i], -5.0, 5.0)
                fitness_new = func(population[i])
                if fitness_new < fitness[i]:
                    fitness[i] = fitness_new
                    if fitness_new < best:
                        best = fitness_new
                        pbest[i] = population[i]
                if fitness_new < func(gbest):
                    gbest = population[i]
            return population, fitness, best, velocity, pbest, gbest

        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best = np.min(fitness)
        pbest = np.copy(population)
        gbest = np.copy(population[np.argmin(fitness)])
        velocity = np.zeros_like(population)

        for _ in range(self.max_iterations):
            population, fitness, best = differential_evolution(population, fitness, best, 0.9, 0.9, 0.3)
            population, fitness, best, velocity, pbest, gbest = particle_swarm_optimization(population, fitness, best, velocity, pbest, gbest)
        return best