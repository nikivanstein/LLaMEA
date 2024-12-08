import numpy as np

class FastFireworksAlgorithmImproved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def local_search(self, func, current_solution):
        candidate_solution = current_solution.copy()
        step_size = 0.1
        for _ in range(10):
            candidate_solution += np.random.uniform(-step_size, step_size, size=self.dim)
            if func(candidate_solution) < func(current_solution):
                current_solution = candidate_solution
        return current_solution

    def __call__(self, func):
        population_size = 10
        fireworks = np.random.uniform(-5.0, 5.0, size=(population_size, self.dim))
        best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
        for _ in range(self.budget // population_size - 1):
            sparks = np.random.uniform(-0.1, 0.1, size=(population_size, self.dim))
            diversity_factor = np.mean(np.std(fireworks, axis=0)
            for i in range(population_size):
                for j in range(self.dim):
                    sparks[i][j] *= diversity_factor * np.abs(best_firework[j] - fireworks[i][j])
            fireworks += sparks
            best_firework = fireworks[np.argmin([func(firework) for firework in fireworks])]
            best_firework = self.local_search(func, best_firework)  # Introducing local search step
        return best_firework