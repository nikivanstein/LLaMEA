import numpy as np

class CustomMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.harmony_memory_size = 5
        self.iterations = 100

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def firefly_move(current, best, attractiveness):
            beta0 = 1.0
            gamma = 0.1
            step_size = gamma * np.linalg.norm(current - best)
            new_position = current + beta0 * np.exp(-attractiveness) * (best - current) + step_size * (np.random.rand(self.dim) - 0.5)
            return np.clip(new_position, -5.0, 5.0)

        def harmony_search():
            hm = np.random.uniform(-5.0, 5.0, (self.harmony_memory_size, self.dim))
            for _ in range(self.iterations):
                harmony_memory_fitness = np.array([objective_function(x) for x in hm])
                best_harmony = hm[np.argmin(harmony_memory_fitness)]
                new_harmony = np.clip(best_harmony + 0.01 * np.random.randn(self.dim), -5.0, 5.0)
                random_index = np.random.randint(self.harmony_memory_size)
                if objective_function(new_harmony) < harmony_memory_fitness[random_index]:
                    hm[random_index] = new_harmony
            return best_harmony

        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        for _ in range(self.budget):
            for _ in range(self.population_size):
                new_solution = firefly_move(best_solution, np.random.uniform(-5.0, 5.0, self.dim), 1.0)
                if objective_function(new_solution) < objective_function(best_solution):
                    best_solution = new_solution
            if np.random.rand() < 0.35:
                best_solution = firefly_move(best_solution, harmony_search(), 0.0)
        return best_solution