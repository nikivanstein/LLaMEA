import numpy as np

class FasterHarmonySearchDynamicMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 10
        harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        mutation_rate = 0.2

        for t in range(1, self.budget + 1):
            new_solution = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            
            # Dynamic Mutation Strategy
            mutation_rate = 0.2 + 0.1 * np.exp(-0.1 * t)  # Adjust mutation rate based on iteration number
            if np.random.rand() < mutation_rate:
                mutation_factor = np.random.uniform(0.1, 0.5, self.dim)
                new_solution = harmony_memory[np.random.randint(population_size)] + mutation_factor * (new_solution - harmony_memory[np.random.randint(population_size)])

            index = np.random.randint(population_size)
            if func(new_solution) < func(harmony_memory[index]):
                harmony_memory[index] = new_solution
            
            # Dynamic Population Size Adaptation
            if t % 10 == 0 and t < self.budget and population_size < 20:
                population_size += 1
                harmony_memory = np.vstack((harmony_memory, np.random.uniform(self.lower_bound, self.upper_bound, (1, self.dim))))

        best_solution = min(harmony_memory, key=func)
        return best_solution