import numpy as np

class FireworkAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def create_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        def explode(firework):
            return firework + np.random.uniform(-1, 1, self.dim)

        fireworks = [create_firework() for _ in range(self.budget)]
        for _ in range(self.budget):
            for i in range(len(fireworks)):
                new_firework = explode(fireworks[i])
                if func(new_firework) < func(fireworks[i]):
                    fireworks[i] = new_firework

        best_solution = fireworks[np.argmin([func(fw) for fw in fireworks])]
        return best_solution