import numpy as np

class DynamicFireworkAlgorithm:
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

        def update_velocity(position, velocity, p_best, g_best, w=0.5, c1=1.5, c2=1.5):
            inertia = w * velocity
            cognitive = c1 * np.random.rand() * (p_best - position)
            social = c2 * np.random.rand() * (g_best - position)
            return inertia + cognitive + social

        fireworks = [create_firework() for _ in range(self.budget)]
        velocities = [np.zeros(self.dim) for _ in range(self.budget)]
        p_best = [fw.copy() for fw in fireworks]
        g_best = fireworks[np.argmin([func(fw) for fw in fireworks])]

        for _ in range(self.budget):
            for i in range(len(fireworks)):
                velocities[i] = update_velocity(fireworks[i], velocities[i], p_best[i], g_best)
                fireworks[i] += velocities[i]

                if func(fireworks[i]) < func(p_best[i]):
                    p_best[i] = fireworks[i]
                if func(fireworks[i]) < func(g_best):
                    g_best = fireworks[i]

                differential_evolution(fireworks, i, func)

        best_solution = fireworks[np.argmin([func(fw) for fw in fireworks])]
        return best_solution