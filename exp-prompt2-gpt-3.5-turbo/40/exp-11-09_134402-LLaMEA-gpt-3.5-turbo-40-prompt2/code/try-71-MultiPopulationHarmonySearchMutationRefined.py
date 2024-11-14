import numpy as np

class MultiPopulationHarmonySearchMutationRefined:
    def __init__(self, budget, dim, num_populations=5, population_size=10, bandwidth=0.01, bandwidth_range=(0.01, 0.1), pitch_adjustment_rate=0.2, pitch_adjustment_range=(0.1, 0.5), memory_consideration_prob=0.5, dynamic_memory_prob_range=(0.4, 0.8), mutation_rate=0.1, opposition_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.population_size = population_size
        self.bandwidth = bandwidth
        self.bandwidth_range = bandwidth_range
        self.pitch_adjustment_rate = pitch_adjustment_rate
        self.pitch_adjustment_range = pitch_adjustment_range
        self.memory_consideration_prob = memory_consideration_prob
        self.dynamic_memory_prob_range = dynamic_memory_prob_range
        self.mutation_rate = mutation_rate
        self.opposition_rate = opposition_rate
        self.population_memories = [np.random.uniform(-5.0, 5.0, (self.population_size, self.dim)) for _ in range(self.num_populations)]

    def __call__(self, func):
        def update_population_memory(population_memory, new_solution):
            population_memory = np.vstack((population_memory, new_solution))
            population_memory = population_memory[np.argsort(func(population_memory))]
            return population_memory[:self.population_size]

        def improvise(population_memory, pop_id):
            new_solution = np.copy(population_memory[np.random.randint(self.population_size)])
            for i in range(self.dim):
                if np.random.rand() < self.bandwidth:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < self.pitch_adjustment_rate:
                    pitch_range = np.random.uniform(*self.pitch_adjustment_range)
                    new_solution[i] += np.random.uniform(-pitch_range, pitch_range)
                    new_solution[i] = np.clip(new_solution[i], -5.0, 5.0)
                if np.random.rand() < np.random.uniform(*self.dynamic_memory_prob_range):
                    new_solution[i] = population_memory[np.random.randint(self.population_size), i]
                if np.random.rand() < self.mutation_rate:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)
                if np.random.rand() < np.random.uniform(0.3, 0.7):
                    new_solution[i] = 2 * np.mean(population_memory[:, i]) - new_solution[i]
            return new_solution

        for _ in range(self.budget):
            for idx, population_memory in enumerate(self.population_memories):
                self.bandwidth = np.clip(self.bandwidth + np.random.uniform(-0.01, 0.01), *self.bandwidth_range)
                self.pitch_adjustment_rate = np.clip(self.pitch_adjustment_rate + np.random.uniform(-0.05, 0.05), *self.pitch_adjustment_range)
                new_solution = improvise(population_memory, idx)
                if func(new_solution) < func(population_memory[-1]):
                    self.population_memories[idx] = update_population_memory(population_memory, new_solution)

        best_solution = np.inf
        for population_memory in self.population_memories:
            if func(population_memory[0]) < best_solution:
                best_solution = population_memory[0]

        return best_solution