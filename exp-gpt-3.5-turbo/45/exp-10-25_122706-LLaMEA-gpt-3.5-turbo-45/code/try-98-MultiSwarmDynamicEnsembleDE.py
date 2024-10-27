import numpy as np

class MultiSwarmDynamicEnsembleDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.swarm_size = 5
        self.F = 0.5
        self.CR = 0.9
        self.min_mut = 0.1
        self.max_mut = 0.9
        self.min_cross = 0.3
        self.max_cross = 0.8
        self.swarms = [np.random.uniform(-5.0, 5.0, (self.swarm_size, dim)) for _ in range(self.pop_size)]

    def __call__(self, func):
        for _ in range(self.budget):
            for swarm in self.swarms:
                trial_swarm = np.zeros_like(swarm)
                for i in range(self.swarm_size):
                    idxs = [idx for idx in range(self.swarm_size) if idx != i]
                    a, b, c = swarm[np.random.choice(idxs, 3, replace=False)]
                    mut = np.clip(np.random.normal(self.F, 0.1), self.min_mut, self.max_mut)
                    cross = np.clip(np.random.normal(self.CR, 0.1), self.min_cross, self.max_cross)
                    trial = a + mut * (b - c)
                    j_rand = np.random.randint(self.dim)
                    for j in range(self.dim):
                        if np.random.rand() < cross or j == j_rand:
                            trial_swarm[i, j] = trial[j]
                        else:
                            trial_swarm[i, j] = swarm[i, j]
                    if func(trial_swarm[i]) < func(swarm[i]):
                        swarm[i] = trial_swarm[i]
                self.F = np.clip(self.F + np.random.normal(0.0, 0.1), self.min_mut, self.max_mut)
                self.CR = np.clip(self.CR + np.random.normal(0.0, 0.1), self.min_cross, self.max_cross)
                if np.random.rand() < 0.45:
                    self.F = np.clip(np.random.normal(self.F, 0.1), self.min_mut, self.max_mut)
                    self.CR = np.clip(np.random.normal(self.CR, 0.1), self.min_cross, self.max_cross)
        best_solution = np.concatenate(self.swarms).T[np.argmin([func(ind) for ind in np.concatenate(self.swarms)])
        return best_solution