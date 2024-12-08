import numpy as np

class ADHSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 30  # Initial Harmony Memory Size
        self.hmcr = 0.95  # Harmony Memory Considering Rate
        self.par = 0.5  # Pitch Adjusting Rate
        self.bw = 0.01 * (self.upper_bound - self.lower_bound)  # Bandwidth
        self.evaluations = 0
        self.f = 0.8  # Differential weight
        self.dynamic_hms_factor = 0.2  # New factor for dynamic memory size

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def pitch_adjustment(self, solution, best_solution):
        diversity_factor = np.std(solution) / self.dim
        adaptive_bw = self.bw * diversity_factor
        for i in range(self.dim):
            if np.random.rand() < self.par:
                adjustment = adaptive_bw * (2 * np.random.rand() - 1)
                solution[i] = np.clip(solution[i] + adjustment, self.lower_bound, self.upper_bound)
        return solution

    def differential_mutation(self, target_idx, harmony_memory):
        idxs = list(range(len(harmony_memory)))  # Changed for dynamic memory size
        idxs.remove(target_idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = harmony_memory[a] + self.f * (harmony_memory[b] - harmony_memory[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        harmony_memory = [self.random_solution() for _ in range(self.hms)]
        harmony_memory_fitness = [func(sol) for sol in harmony_memory]
        self.evaluations += self.hms

        best_solution = harmony_memory[np.argmin(harmony_memory_fitness)]
        best_fitness = min(harmony_memory_fitness)

        while self.evaluations < self.budget:
            new_solution = np.zeros(self.dim)
            # Adjust harmony memory size dynamically
            self.hms = int(self.hms + self.dynamic_hms_factor * np.sin(np.pi * self.evaluations / self.budget))

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = harmony_memory[np.random.randint(len(harmony_memory))]
                    new_solution[i] = selected_harmony[i]
                else:
                    new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)

            new_solution = self.pitch_adjustment(new_solution, best_solution)
            new_solution = self.differential_mutation(np.argmin(harmony_memory_fitness), harmony_memory)

            new_solution_fitness = func(new_solution)
            self.evaluations += 1

            if new_solution_fitness < max(harmony_memory_fitness):
                worst_index = np.argmax(harmony_memory_fitness)
                harmony_memory[worst_index] = new_solution
                harmony_memory_fitness[worst_index] = new_solution_fitness

                if new_solution_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_solution_fitness

        return best_solution