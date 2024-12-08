import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DynamicBandwidthHarmonySearchImproved:
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
                    new_harmony[i] = np.random.uniform(self.lower_bound, self.upper_bound)
            return new_harmony

        def update_bandwidth(bandwidth, convergence_speed):
            return bandwidth * (1 - np.exp(-0.1 * convergence_speed))

        harmony_memory = [generate_random_solution() for _ in range(10)]
        bandwidth = 0.9
        bandwidth_decay_rate = 0.95
        convergence_speed = 0.0
        population_size = 10  # Dynamic population size
        with ThreadPoolExecutor() as executor:
            for itr in range(self.budget):
                candidates = [improvise(harmony_memory, bandwidth) for _ in range(len(harmony_memory))]
                results = list(executor.map(func, candidates))
                for idx, result in enumerate(results):
                    if result < func(harmony_memory[-1]):
                        harmony_memory[-1] = candidates[idx]
                        harmony_memory.sort(key=func)
                if itr % 10 == 0:
                    bandwidth = update_bandwidth(bandwidth, convergence_speed)
                    population_size = min(20, int(10 + itr/100))  # Adjust population size dynamically
                    harmony_memory = harmony_memory[:population_size] + [generate_random_solution() for _ in range(population_size - len(harmony_memory))]
                    convergence_speed = np.mean(np.abs(np.diff([func(sol) for sol in harmony_memory])))
        return harmony_memory[0]