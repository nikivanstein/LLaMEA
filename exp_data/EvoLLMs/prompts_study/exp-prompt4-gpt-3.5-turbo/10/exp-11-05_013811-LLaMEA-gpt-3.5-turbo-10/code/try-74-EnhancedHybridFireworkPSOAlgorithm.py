import numpy as np

class EnhancedHybridFireworkPSOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.n_fireworks = 10
        self.n_sparks = 5
        self.mutation_scale = 1.0
        self.inertia_weight = 0.5
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.de_weight = 0.7
        self.de_cr = 0.9

    def __call__(self, func):
        def create_firework():
            return np.random.uniform(-5.0, 5.0, self.dim)

        fireworks = [create_firework() for _ in range(self.n_fireworks)]
        best_firework = min(fireworks, key=lambda x: func(x))

        for _ in range(self.budget - self.n_fireworks):
            new_fireworks = [firework + np.random.normal(0, self.mutation_scale, self.dim) for firework in fireworks]
            new_sparks = [firework + np.random.normal(0, self.mutation_scale * 0.2, self.dim) for _ in range(self.n_sparks) for firework in fireworks]
            
            # Differential Evolution Strategy
            for i in range(self.n_fireworks):
                mutant = fireworks[np.random.choice(range(self.n_fireworks))]
                trial = fireworks[i] + self.de_weight * (mutant - fireworks[i])
                for j in range(self.dim):
                    if np.random.rand() > self.de_cr:
                        trial[j] = mutant[j]
                if func(trial) < func(fireworks[i]):
                    fireworks[i] = trial
            
            fireworks += new_sparks

            velocities = [np.zeros(self.dim) for _ in range(self.n_fireworks)]
            global_best = fireworks[0]
            for i in range(self.n_fireworks):
                velocities[i] = self.inertia_weight * velocities[i] + self.cognitive_weight * np.random.rand() * (best_firework - fireworks[i]) + self.social_weight * np.random.rand() * (global_best - fireworks[i])
                fireworks[i] += velocities[i]

            fireworks.sort(key=lambda x: func(x))
            fireworks = fireworks[:self.n_fireworks]
            if func(fireworks[0]) < func(best_firework):
                best_firework = fireworks[0]
            self.mutation_scale = max(0.1, self.mutation_scale * 0.99)

            if np.random.rand() < 0.1:
                self.n_fireworks = max(5, int(self.n_fireworks * 1.1)) if np.random.rand() < 0.5 else max(5, int(self.n_fireworks * 0.9))

        return best_firework