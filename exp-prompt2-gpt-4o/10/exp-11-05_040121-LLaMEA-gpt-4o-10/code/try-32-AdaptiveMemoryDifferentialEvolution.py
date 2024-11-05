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
        success_rate = 0.0

        while evaluations < self.budget:
            successful_mutations = 0
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
                    successful_mutations += 1

                if evaluations >= self.budget:
                    break

            success_rate = successful_mutations / population_size
            if evaluations < self.budget and success_rate < 0.2:
                mem_F[mem_idx] = mem_F[mem_idx] * 1.1
                mem_CR[mem_idx] = max(0.1, mem_CR[mem_idx] * 0.9)

            if evaluations < self.budget and evaluations % (population_size * 10) == 0:
                population = np.where(np.random.rand(population_size, self.dim) < 0.1,
                                      np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim)),
                                      population)
                fitness = np.array([func(ind) for ind in population])
                evaluations += population_size

            # Dynamic population resizing
            if evaluations < self.budget and evaluations % (population_size * 5) == 0:
                population_size = max(4, int(population_size * 0.9))
                population = population[:population_size]
                fitness = fitness[:population_size]

        return population[np.argmin(fitness)], min(fitness)