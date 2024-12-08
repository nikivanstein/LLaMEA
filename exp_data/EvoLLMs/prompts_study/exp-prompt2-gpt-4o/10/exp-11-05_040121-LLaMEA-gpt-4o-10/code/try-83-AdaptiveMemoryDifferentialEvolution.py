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
        inertia_weight = 0.9  # Initial inertia weight
        decay_rate = 0.99  # Inertia weight decay
        chaotic_seq = np.random.rand(population_size)  # Chaotic sequence initialization

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
                chaotic_seq = np.mod(4.0 * chaotic_seq * (1.0 - chaotic_seq), 1.0)  # Update chaotic sequence
                population = np.where(np.random.rand(population_size, self.dim) < chaotic_seq[:, None],
                                      np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim)),
                                      population)
                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

            # Adaptive population resizing and inertia application
            if evaluations < self.budget and evaluations % (population_size * 5) == 0:
                population_size = max(4, int(population_size * 0.85))
                population = population[:population_size]
                fitness = fitness[:population_size]
                inertia_weight *= decay_rate  # Apply inertia weight decay
                population += np.random.normal(0, inertia_weight, population.shape)  # Apply inertia weight effect

        return population[np.argmin(fitness)], min(fitness)