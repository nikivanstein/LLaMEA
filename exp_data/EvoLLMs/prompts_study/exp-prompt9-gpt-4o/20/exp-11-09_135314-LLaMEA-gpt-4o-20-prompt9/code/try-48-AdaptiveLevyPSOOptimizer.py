import numpy as np
from scipy.optimize import minimize

class AdaptiveLevyPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(40, budget // 10)
        self.strategy_switch = 0.25  # Switch to PSO after 25% of budget

    def levy_flight(self, L=1.5):
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / L))
        return step * 0.01

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocity = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evals = self.population_size
        pbest = population.copy()
        pbest_fitness = fitness.copy()
        best_idx = np.argmin(fitness)
        gbest = population[best_idx].copy()
        gbest_fitness = fitness[best_idx]

        while evals < self.budget:
            if evals < self.strategy_switch * self.budget:
                # Lévy flight for exploration
                for i in range(self.population_size):
                    candidate = population[i] + self.levy_flight()
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evals += 1
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                        if candidate_fitness < gbest_fitness:
                            gbest_fitness = candidate_fitness
                            gbest = candidate.copy()
                    if evals >= self.budget:
                        break
            else:
                # Particle Swarm Optimization
                for i in range(self.population_size):
                    inertia_weight = 0.729
                    cognitive_const = 1.49445 * np.random.rand(self.dim)
                    social_const = 1.49445 * np.random.rand(self.dim)
                    velocity[i] = (inertia_weight * velocity[i] +
                                   cognitive_const * (pbest[i] - population[i]) +
                                   social_const * (gbest - population[i]))
                    population[i] += velocity[i]
                    population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)
                    new_fitness = func(population[i])
                    evals += 1
                    if new_fitness < pbest_fitness[i]:
                        pbest[i] = population[i].copy()
                        pbest_fitness[i] = new_fitness
                        if new_fitness < gbest_fitness:
                            gbest_fitness = new_fitness
                            gbest = population[i].copy()
                    if evals >= self.budget:
                        break

        return gbest