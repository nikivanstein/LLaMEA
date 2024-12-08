import numpy as np

class EnhancedADHSOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.hms = 30  # Harmony Memory Size
        self.hmcr = 0.9 + 0.1 * np.random.rand()  # Adaptive HMCR
        self.par = 0.5  # Pitch Adjusting Rate
        self.bw = 0.01 * (self.upper_bound - self.lower_bound)  # Bandwidth
        self.evaluations = 0
        self.f = 0.8  # Differential weight
        self.cr = 0.9  # Crossover probability
        self.learning_rate = 0.05  # Learning rate for parameter adjustment

    def random_solution(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)

    def pitch_adjustment(self, solution, best_solution):
        diversity_factor = np.std(solution) / self.dim
        adaptive_bw = self.bw * diversity_factor
        for i in range(self.dim):
            if np.random.rand() < self.par:
                adjustment = adaptive_bw * (2 * np.random.rand() - 1)
                solution[i] = np.clip(solution[i] + adjustment, self.lower_bound, self.upper_bound)
        self.par = np.clip(self.par + self.learning_rate * np.random.rand(), 0.1, 0.9)  # Adjusting PAR dynamically
        return solution

    def differential_mutation(self, target_idx, harmony_memory):
        idxs = list(range(self.hms))
        idxs.remove(target_idx)
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = harmony_memory[a] + self.f * (harmony_memory[b] - harmony_memory[c])
        mutant = np.where(np.random.rand(self.dim) < self.cr, mutant, harmony_memory[target_idx])
        self.f = np.clip(self.f + self.learning_rate * (best_solution - mutant).mean(), 0.4, 1.2)  # Dynamic adjustment of 'f'
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        harmony_memory = [self.random_solution() for _ in range(self.hms)]
        harmony_memory_fitness = [func(sol) for sol in harmony_memory]
        self.evaluations += self.hms

        best_solution = harmony_memory[np.argmin(harmony_memory_fitness)]
        best_fitness = min(harmony_memory_fitness)

        while self.evaluations < self.budget:
            new_solution = np.zeros(self.dim)

            for i in range(self.dim):
                if np.random.rand() < self.hmcr:
                    selected_harmony = harmony_memory[np.random.randint(self.hms)]
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