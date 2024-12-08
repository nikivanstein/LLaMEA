import numpy as np

class EnhancedFireworkAlgorithmOpposition:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def create_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def explode(firework):
            return firework + np.random.uniform(-1, 1, self.dim)

        def differential_evolution(fireworks, i, func):
            F = 0.5
            for j in range(len(fireworks)):
                if j != i:
                    r1, r2, r3 = np.random.choice(len(fireworks), 3, replace=False)
                    mutant = fireworks[r1] + F * (fireworks[r2] - fireworks[r3])
                    if func(mutant) < func(fireworks[j]):
                        fireworks[j] = mutant

        fireworks = [create_firework() for _ in range(self.budget)]
        for _ in range(self.budget):
            for i in range(len(fireworks)):
                new_firework = explode(fireworks[i])
                if func(new_firework) < func(fireworks[i]):
                    fireworks[i] = new_firework
                differential_evolution(fireworks, i, func)

        # Incorporating opposition-based learning
        best_solution = fireworks[np.argmin([func(fw) for fw in fireworks])]
        opposite_solution = -best_solution
        if func(opposite_solution) < func(best_solution):
            best_solution = opposite_solution

        return best_solution