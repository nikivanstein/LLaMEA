import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=20, mutation_factor=0.5, crossover_prob=0.9, inertia_weight=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.inertia_weight = inertia_weight

    def __call__(self, func):
        def pso_de(func):
            population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
            fitness = np.array([func(ind) for ind in population])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            velocity = np.zeros((self.swarm_size, self.dim))

            for _ in range(self.budget - self.swarm_size):
                for i in range(self.swarm_size):
                    p_best = population[np.argmin(fitness)]

                    r1, r2 = np.random.uniform(0, 1, 2)
                    velocity[i] = self.inertia_weight * velocity[i] + r1 * self.mutation_factor * (p_best - population[i]) + r2 * (best_solution - population[i])
                    population[i] += velocity[i]

                    if np.random.uniform(0, 1) < self.crossover_prob:
                        candidate = population[np.random.choice(range(self.swarm_size), 3, replace=False)]
                        trial_vector = population[i] + self.mutation_factor * (candidate[0] - candidate[1])
                        trial_vector[candidate[2] < 0.5] = candidate[2]

                        if func(trial_vector) < fitness[i]:
                            population[i] = trial_vector
                            fitness[i] = func(trial_vector)

                best_idx = np.argmin(fitness)
                best_solution = population[best_idx]

            return best_solution

        return pso_de(func)