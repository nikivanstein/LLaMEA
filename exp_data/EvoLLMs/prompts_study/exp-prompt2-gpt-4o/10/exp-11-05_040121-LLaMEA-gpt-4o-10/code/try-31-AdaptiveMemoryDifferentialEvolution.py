import numpy as np
import scipy.stats as stats

class AdaptiveMemoryDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def levy_flight(self, L):
        beta = 1.5
        sigma_u = (stats.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (stats.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=L)
        v = np.random.normal(0, 1, size=L)
        step = u / np.abs(v) ** (1 / beta)
        return step

    def __call__(self, func):
        np.random.seed(42)
        population_size = 10 * self.dim
        F = 0.8
        CR = 0.9
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size
        memory_size = 5
        mem_F = np.full(memory_size, F)
        mem_CR = np.full(memory_size, CR)
        mem_idx = 0

        while evaluations < self.budget:
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                F = mem_F[np.random.randint(memory_size)]
                CR = mem_CR[np.random.randint(memory_size)]
                mutated = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.array([mutated[j] if np.random.rand() < CR else population[i][j] for j in range(self.dim)])
                new_fitness = func(crossover)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = crossover
                    fitness[i] = new_fitness
                    mem_F[mem_idx] = F
                    mem_CR[mem_idx] = CR
                    mem_idx = (mem_idx + 1) % memory_size

                if evaluations >= self.budget:
                    break

            if evaluations < self.budget and evaluations % (population_size * 10) == 0:
                population = np.where(np.random.rand(population_size, self.dim) < 0.1,
                                      np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim)),
                                      population)
                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

            if evaluations < self.budget and evaluations % (population_size * 5) == 0:
                population_size = max(4, int(population_size * 0.9))
                population = population[:population_size]
                fitness = fitness[:population_size]

            # Local search with Lévy flights
            if evaluations < self.budget and np.random.rand() < 0.3:
                step = self.levy_flight(self.dim)
                for i in range(population_size):
                    candidate = population[i] + step
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = candidate_fitness
                    if evaluations >= self.budget:
                        break

        return population[np.argmin(fitness)], min(fitness)