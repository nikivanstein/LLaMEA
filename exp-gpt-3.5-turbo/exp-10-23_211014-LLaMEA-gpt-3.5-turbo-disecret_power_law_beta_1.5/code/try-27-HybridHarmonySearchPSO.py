import numpy as np

class HybridHarmonySearchPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.harmony_memory_size = 20
        self.pso_c1 = 1.49
        self.pso_c2 = 1.49
        self.pso_w = 0.72

    def _harmony_search(self, population, func):
        harmony_memory = np.random.uniform(-5.0, 5.0, size=(self.harmony_memory_size, self.dim))
        harmony_fitness = np.array([func(individual) for individual in harmony_memory])
        for _ in range(self.population_size):
            new_harmony = np.clip(np.mean(harmony_memory, axis=0) + np.random.uniform(-1, 1, self.dim), -5.0, 5.0)
            if func(new_harmony) < max(harmony_fitness):
                replace_idx = np.argmax(harmony_fitness)
                harmony_memory[replace_idx] = new_harmony
                harmony_fitness[replace_idx] = func(new_harmony)
        return harmony_memory[np.argmin(harmony_fitness)]

    def _particle_swarm_optimization(self, population, func):
        velocity = np.zeros((self.population_size, self.dim))
        pbest = population.copy()
        pbest_fitness = np.array([func(individual) for individual in population])
        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx]

        for _ in range(self.population_size):
            velocity = self.pso_w * velocity + self.pso_c1 * np.random.rand() * (pbest - population) + self.pso_c2 * np.random.rand() * (gbest - population)
            population += velocity
            population = np.clip(population, -5.0, 5.0)
            current_fitness = np.array([func(individual) for individual in population])

            for i in range(self.population_size):
                if current_fitness[i] < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = current_fitness[i]

            new_gbest_idx = np.argmin(pbest_fitness)
            if pbest_fitness[new_gbest_idx] < func(gbest):
                gbest = pbest[new_gbest_idx]
                gbest_idx = new_gbest_idx

        return gbest

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))

        for _ in range(self.budget // self.population_size):
            population = self._harmony_search(population, func)
            population = self._particle_swarm_optimization(population, func)

        best_solution = self._particle_swarm_optimization(population, func)
        return best_solution