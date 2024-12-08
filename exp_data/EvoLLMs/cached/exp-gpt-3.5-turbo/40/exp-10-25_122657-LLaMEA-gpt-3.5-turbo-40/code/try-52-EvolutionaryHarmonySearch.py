import numpy as np

class EvolutionaryHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))

        def get_fitness(harmony_memory):
            return np.array([func(solution) for solution in harmony_memory])

        def update_harmony_memory(harmony_memory, fitness):
            worst_idx = np.argmax(fitness)
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            harmony_memory[worst_idx] = new_solution
            return harmony_memory

        def evolutionary_step(harmony_memory, fitness):
            best_idx = np.argmin(fitness)
            population_mean = np.mean(harmony_memory, axis=0)
            offspring = np.random.normal(population_mean, np.std(harmony_memory), (self.budget, self.dim))
            offspring_fitness = get_fitness(offspring)

            replace_idx = np.argmax(offspring_fitness)
            harmony_memory[replace_idx] = offspring[replace_idx]

            return harmony_memory

        harmony_memory = initialize_harmony_memory()
        fitness = get_fitness(harmony_memory)

        for _ in range(self.budget - self.budget // 10):
            harmony_memory = update_harmony_memory(harmony_memory, fitness)
            fitness = get_fitness(harmony_memory)
            harmony_memory = evolutionary_step(harmony_memory, fitness)

        best_idx = np.argmin(fitness)
        return harmony_memory[best_idx]