import numpy as np

class HybridGreyWolfPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.budget, self.dim))

        def grey_wolf_optimizer(population):
            alpha, beta, delta = population[np.argsort([func(ind) for ind in population])[:3]]
            a = 2 - 2 * (_ / self.budget)
            for i in range(self.budget):
                x = population[i]
                X1 = alpha - a * np.abs(2 * np.random.rand(self.dim) * alpha - x)
                X2 = beta - a * np.abs(2 * np.random.rand(self.dim) * beta - x)
                X3 = delta - a * np.abs(2 * np.random.rand(self.dim) * delta - x)
                population[i] = (X1 + X2 + X3) / 3
            return population

        def particle_swarm_optimizer(population):
            inertia_weight = 0.5
            cognitive_weight = 0.5
            social_weight = 0.5
            best_particle = population[np.argmin([func(ind) for ind in population])]
            for i in range(self.budget):
                for j in range(self.dim):
                    velocity = inertia_weight * population[i][j] + cognitive_weight * np.random.rand() * (best_particle[j] - population[i][j]) + social_weight * np.random.rand() * (population[i][j] - best_particle[j])
                    population[i][j] += velocity
            return population

        population = initialize_population()
        for _ in range(self.budget):
            population = grey_wolf_optimizer(population)
            population = particle_swarm_optimizer(population)

        return population[np.argmin([func(ind) for ind in population])]
        
        return optimize()
