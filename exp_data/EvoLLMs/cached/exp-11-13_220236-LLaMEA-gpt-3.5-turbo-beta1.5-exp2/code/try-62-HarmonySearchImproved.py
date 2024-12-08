import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HarmonySearchImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def generate_random_solution():
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

        def improvise(harmony_memory, bandwidth):
            new_harmony = np.copy(harmony_memory[np.random.randint(0, len(harmony_memory))])
            for i in range(self.dim):
                if np.random.rand() < bandwidth:
                    mutation_step = np.random.uniform(0.1, 1.0)  # Adaptive mutation step size
                    new_harmony[i] += mutation_step * np.random.uniform(-1, 1)
                    new_harmony[i] = np.clip(new_harmony[i], self.lower_bound, self.upper_bound)
            return new_harmony

        harmony_memory = [generate_random_solution() for _ in range(10)]
        bandwidth = 0.9
        bandwidth_decay_rate = 0.95
        population_size = 10  # Dynamic population size
        best_solution = harmony_memory[0] # Initialize best solution
        best_fitness = func(best_solution) # Evaluate best solution
        with ThreadPoolExecutor() as executor:
            for itr in range(self.budget):
                candidates = [improvise(harmony_memory, bandwidth) for _ in range(len(harmony_memory))]
                results = list(executor.map(func, candidates))  # Evaluate candidates in parallel
                for idx, result in enumerate(results):
                    if result < func(harmony_memory[-1]):
                        harmony_memory[-1] = candidates[idx]
                        harmony_memory.sort(key=func)
                if results[0] < best_fitness: # Check if the new solution is better
                    best_solution = harmony_memory[0] # Update best solution
                    best_fitness = results[0] # Update best fitness
                if itr % 10 == 0:
                    bandwidth *= bandwidth_decay_rate
                    population_size = min(20, int(10 + itr/100))  # Adjust population size dynamically
                    harmony_memory = harmony_memory[:population_size] + [generate_random_solution() for _ in range(population_size - len(harmony_memory))]
        return best_solution