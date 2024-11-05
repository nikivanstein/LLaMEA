import numpy as np

class AdaptiveMemoryDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
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
        inertia_weight = 0.9  # New inertia weight

        while evaluations < self.budget:
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Dynamically adjust F and CR for exploration
                F = np.random.uniform(0.5, 1.0)
                CR = np.random.uniform(0.7, 1.0)
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
                population = np.where(np.random.rand(population_size, self.dim) < 0.3,
                                      np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim)),
                                      population)
                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

            # Adaptive population resizing and inertia application
            if evaluations < self.budget and evaluations % (population_size * 5) == 0:
                population_size = max(4, int(population_size * 0.85))
                population = population[:population_size]
                fitness = fitness[:population_size]
                population += np.random.normal(0, inertia_weight, population.shape)

            # Environment-driven local search to enhance exploration-exploitation balance
            if evaluations < self.budget and evaluations % (population_size * 3) == 0:
                local_search_radius = 0.1
                for j in range(population_size):
                    local_variation = np.random.uniform(-local_search_radius, local_search_radius, self.dim)
                    candidate = np.clip(population[j] + local_variation, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < fitness[j]:
                        population[j] = candidate
                        fitness[j] = candidate_fitness

        return population[np.argmin(fitness)], min(fitness)