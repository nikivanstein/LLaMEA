import numpy as np
from concurrent.futures import ThreadPoolExecutor

class HarmonySearchParallel:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hmcr = 0.7
        self.par = 0.4
        self.bandwidth = 0.01
        self.hm_size = int(0.5 * budget)
        self.num_threads = 4  # Number of threads for parallel evaluation

    def evaluate_solution(self, func, solution):
        return func(solution)

    def __call__(self, func):
        def initialize_harmony_memory():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.hm_size, self.dim))

        def pitch_adjustment(new_solution, index):
            if np.random.rand() < self.par:
                new_solution[index] = new_solution[index] + np.random.uniform(-self.bandwidth, self.bandwidth)
                new_solution[index] = np.clip(new_solution[index], self.lower_bound, self.upper_bound)
            return new_solution

        harmony_memory = initialize_harmony_memory()
        fitness = np.array([func(solution) for solution in harmony_memory])
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for _ in range(self.budget - self.hm_size):
                new_solutions = []
                for _ in range(self.num_threads):
                    new_solution = np.zeros(self.dim)
                    for i in range(self.dim):
                        if np.random.rand() < self.hmcr:
                            new_solution[i] = harmony_memory[np.random.choice(self.hm_size)][i]
                        else:
                            new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)
                        new_solution = pitch_adjustment(new_solution, i)
                    new_solutions.append(new_solution)
                
                # Evaluate new solutions in parallel
                new_fitness = np.array(list(executor.map(lambda x: self.evaluate_solution(func, x), new_solutions)))

                for idx, new_solution in enumerate(new_solutions):
                    worst_index = np.argmax(fitness)
                    if new_fitness[idx] < fitness[worst_index]:
                        harmony_memory[worst_index] = new_solution
                        fitness[worst_index] = new_fitness[idx]

        best_index = np.argmin(fitness)
        return harmony_memory[best_index], fitness[best_index]