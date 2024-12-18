import numpy as np

class EnhancedDynamicHarmonySearch(DynamicHarmonySearch):
    def __init__(self, budget, dim, local_search_radius=0.1):
        super().__init__(budget, dim)
        self.local_search_radius = local_search_radius

    def local_search(self, solution, func):
        neighborhood = np.random.uniform(-self.local_search_radius, self.local_search_radius, size=(self.dim,))
        new_solution = solution + neighborhood
        new_solution = np.clip(new_solution, -5.0, 5.0)
        return new_solution if func(new_solution) < func(solution) else solution

    def __call__(self, func):
        harmony_memory = self.initialize_harmony_memory()
        best_solution = None
        best_fitness = np.inf

        for iteration in range(self.budget):
            par, bandwidth = self.update_parameters(iteration)
            new_harmony = self.improvise_harmony(harmony_memory, par, bandwidth)
            new_harmony = self.local_search(new_harmony, func)
            new_fitness = func(new_harmony)

            if new_fitness < best_fitness:
                best_solution = new_harmony
                best_fitness = new_fitness

            idx = np.argmax([func(h) for h in harmony_memory])
            if new_fitness < func(harmony_memory[idx]):
                harmony_memory[idx] = new_harmony

        return best_solution