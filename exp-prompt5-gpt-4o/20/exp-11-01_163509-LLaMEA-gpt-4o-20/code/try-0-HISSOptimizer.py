import numpy as np

class HISSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 30  # Harmony Memory Size
        self.hmcr = 0.95  # Harmony Memory Considering Rate
        self.par = 0.5  # Pitch Adjusting Rate
        self.bw = 0.01 * (self.upper_bound - self.lower_bound)  # Bandwidth
        self.evaluations = 0

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def pitch_adjustment(self, solution):
        for i in range(self.dim):
            if np.random.rand() < self.par:
                adjustment = self.bw * (2 * np.random.rand() - 1)
                solution[i] = np.clip(solution[i] + adjustment, self.lower_bound, self.upper_bound)
        return solution

    def __call__(self, func):
        # Initialize harmony memory with random solutions
        harmony_memory = [self.random_solution() for _ in range(self.hms)]
        harmony_memory_fitness = [func(sol) for sol in harmony_memory]
        self.evaluations += self.hms

        best_solution = harmony_memory[np.argmin(harmony_memory_fitness)]
        best_fitness = min(harmony_memory_fitness)

        while self.evaluations < self.budget:
            new_solution = np.zeros(self.dim)

            # Generate new solution based on harmony memory
            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = harmony_memory[np.random.randint(self.hms)]
                    new_solution[i] = selected_harmony[i]
                else:
                    new_solution[i] = np.random.uniform(self.lower_bound, self.upper_bound)

            # Pitch adjustment
            new_solution = self.pitch_adjustment(new_solution)

            # Evaluate new solution
            new_solution_fitness = func(new_solution)
            self.evaluations += 1

            # Update harmony memory if the new solution is better
            if new_solution_fitness < max(harmony_memory_fitness):
                worst_index = np.argmax(harmony_memory_fitness)
                harmony_memory[worst_index] = new_solution
                harmony_memory_fitness[worst_index] = new_solution_fitness

                # Update the best solution found so far
                if new_solution_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_solution_fitness

        return best_solution

# Example usage:
# optimizer = HISSOptimizer(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)