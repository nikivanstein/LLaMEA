import numpy as np
import random

class MultiSwarmHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.persists = 0.35  # persistence probability
        self.q = 0.5  # q parameter
        self.rho = 0.5  # rho parameter
        self.w = 0.5  # w parameter
        self.pbest = np.zeros((self.dim,))
        self.gbest = np.zeros((self.dim,))
        self.swarm = np.random.uniform(-5.0, 5.0, (self.dim, self.budget))

    def __call__(self, func):
        for i in range(self.budget):
            # calculate fitness for each solution in the swarm
            fitness = func(self.swarm[:, i])

            # update pbest for each solution
            self.pbest = np.minimum(self.pbest, self.swarm[:, i])

            # calculate harmony mean
            harmony_mean = np.mean(fitness)

            # calculate best harmony
            best_harmony = np.min(fitness)

            # update gbest if best harmony is found
            if best_harmony < np.min(self.gbest):
                self.gbest = np.copy(self.swarm[:, i])
                self.w = self.q

            # update persistence probability
            if np.random.rand() < self.persists:
                # randomly select a solution to refine
                refine_idx = np.random.choice(self.budget)
                # perturb the selected solution
                self.swarm[:, refine_idx] = self.swarm[:, refine_idx] + np.random.uniform(-0.1, 0.1, self.dim)
                # calculate fitness for the refined solution
                fitness = func(self.swarm[:, refine_idx])
                # update pbest for the refined solution
                self.pbest = np.minimum(self.pbest, self.swarm[:, refine_idx])
                # calculate harmony mean for the refined solution
                harmony_mean = np.mean(fitness)
                # calculate best harmony for the refined solution
                best_harmony = np.min(fitness)
                # update gbest if best harmony is found for the refined solution
                if best_harmony < np.min(self.gbest):
                    self.gbest = np.copy(self.swarm[:, refine_idx])
                    self.w = self.q

            # update swarm using reflection operator
            elif self.w < 0.5:
                # reflect solutions in the swarm
                self.swarm[:, i] = 2 * self.swarm[:, i] - self.swarm[:, i]
                # calculate fitness for each solution in the swarm
                fitness = func(self.swarm[:, i])
                # update pbest for each solution
                self.pbest = np.minimum(self.pbest, self.swarm[:, i])
                # calculate harmony mean
                harmony_mean = np.mean(fitness)
                # calculate best harmony
                best_harmony = np.min(fitness)
                # update gbest if best harmony is found
                if best_harmony < np.min(self.gbest):
                    self.gbest = np.copy(self.swarm[:, i])
                    self.w = self.q

# Example usage:
def func(x):
    return np.sum(x**2)

ms = MultiSwarmHarmonySearch(100, 2)
best_x = ms(func)
print("Best x:", best_x)
print("Best f(x):", func(best_x))