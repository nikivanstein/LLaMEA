import numpy as np

class HarmonySearchDifferentialEvolutionAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.harmony_memory_size = 5
        self.p_accept = 0.7
        self.bandwidth = 0.01
        self.f = 0.5

    def __call__(self, func):
        def generate_new_solution(bounds):
            return np.random.uniform(bounds[0], bounds[1], self.dim)

        def clipToBounds(solution, bounds):
            return np.clip(solution, bounds[0], bounds[1])

        def objective_function(solution):
            return func(solution)

        def harmony_search(bounds):
            harmonies = [generate_new_solution(bounds) for _ in range(self.population_size)]

            for _ in range(self.budget):
                new_harmony = generate_new_solution(bounds)
                for j in range(self.dim):
                    if np.random.rand() < self.p_accept:
                        new_harmony[j] = harmonies[np.random.randint(self.population_size)][j]
                        if np.random.rand() < self.bandwidth:
                            new_harmony[j] += self.f * (harmonies[np.random.randint(self.population_size)][j] - harmonies[np.random.randint(self.population_size)][j])

                new_harmony = clipToBounds(new_harmony, bounds)
                if objective_function(new_harmony) < min([objective_function(h) for h in harmonies]):
                    harmonies[np.argmax([objective_function(h) for h in harmonies])] = new_harmony

            return min(harmonies, key=objective_function)

        search_space = [(-5.0, 5.0) for _ in range(self.dim)]
        return harmony_search(search_space)