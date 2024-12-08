import numpy as np

class DynamicMutationEnhancedARS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.alpha = 1.2
        self.beta = 0.8
        self.initial_mutation_rate = 0.1
        self.mutation_rate = self.initial_mutation_rate

    def local_search(self, center, func):
        neighborhood = 0.1
        new_center = center.copy()
        for i in range(self.dim):
            new_center[i] += np.random.uniform(-neighborhood, neighborhood)
            new_center[i] = np.clip(new_center[i], -5.0, 5.0)

        if func(new_center) < func(center):
            return new_center
        else:
            return center

    def adjust_mutation_rate(self, current_fitness, prev_fitness):
        if current_fitness < prev_fitness:
            self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
        else:
            self.mutation_rate = max(0.01, self.mutation_rate * 0.8)

    def initialize_population(self, num_solutions, dim):
        initial_population = []
        for _ in range(num_solutions):
            candidate = np.random.uniform(-5.0, 5.0, dim)
            for i in range(dim):
                if np.random.rand() < self.mutation_rate:
                    candidate[i] = np.random.uniform(-5.0, 5.0)
            initial_population.append(candidate)
        return initial_population

    def __call__(self, func):
        initial_solutions = self.initialize_population(5, self.dim)
        best_solution = min(initial_solutions, key=lambda x: func(x))
        best_fitness = func(best_solution)
        step_size = 1.0

        for _ in range(self.budget):
            candidate_solution = best_solution + step_size * np.random.uniform(-1, 1, self.dim)
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)
            candidate_solution = self.local_search(candidate_solution, func)
            candidate_fitness = func(candidate_solution)

            self.adjust_mutation_rate(candidate_fitness, best_fitness)

            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                step_size *= self.alpha
            else:
                step_size *= self.beta

        return best_solution