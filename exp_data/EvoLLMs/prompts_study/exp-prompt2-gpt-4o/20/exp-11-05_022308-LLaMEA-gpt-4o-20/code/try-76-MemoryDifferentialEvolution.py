import numpy as np

class MemoryDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f = 0.8
        self.cr = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.memory_size = 5  # Introduced memory for past solutions
        self.reinit_threshold = budget // 8

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size
        eval_step = self.budget // (3 * self.pop_size)
        no_improvement_count = 0
        memory = []  # Memory to store past solutions

        while evals < self.budget:
            best_before = np.min(fitness)
            for i in range(self.pop_size):
                self.f = 0.5 + 0.3 * np.sin(2 * np.pi * evals / self.budget + np.max(fitness))
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = population[indices]
                if memory and np.random.rand() < 0.5:  # Stochastic use of memory
                    a = memory[np.random.randint(len(memory))]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)

                self.cr = 0.6 + 0.4 * np.random.rand()  # Stochastic crossover rate
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    memory.append(trial)  # Update memory
                    if len(memory) > self.memory_size:
                        memory.pop(0)  # Maintain memory size

                if evals >= self.budget:
                    break

            if evals % eval_step == 0:
                best_idx = np.argmin(fitness)
                local_search_vector = population[best_idx] + np.random.normal(0, 0.05, self.dim)
                local_search_vector = np.clip(local_search_vector, self.lower_bound, self.upper_bound)
                local_fitness = func(local_search_vector)
                evals += 1
                if local_fitness < fitness[best_idx]:
                    population[best_idx] = local_search_vector
                    fitness[best_idx] = local_fitness

            if np.min(fitness) == best_before:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= self.reinit_threshold:
                worst_indices = fitness.argsort()[-self.pop_size//4:]
                population[worst_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(worst_indices), self.dim))
                fitness[worst_indices] = np.array([func(ind) for ind in population[worst_indices]])
                evals += len(worst_indices)
                no_improvement_count = 0

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]