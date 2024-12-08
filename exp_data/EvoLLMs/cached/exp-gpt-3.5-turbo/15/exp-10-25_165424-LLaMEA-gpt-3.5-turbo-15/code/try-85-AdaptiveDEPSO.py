import numpy as np

class AdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.cr = 0.8
        self.f = 0.6

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best = pop[best_idx].copy()

        evals = self.pop_size
        while evals < self.budget:
            new_pop = np.zeros_like(pop)
            for i in range(self.pop_size):
                trial = pop[i].copy()
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)

                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == np.random.randint(0, self.dim):
                        trial[j] = pop[r1, j] + self.f * (pop[r2, j] - pop[r3, j])
                trial = np.clip(trial, lower_bound, upper_bound)
                
                for j in range(self.dim):
                    if np.random.rand() < self.w:
                        trial[j] += self.c1 * np.random.rand() * (best[j] - pop[i, j])
                        trial[j] += self.c2 * np.random.rand() * (pop[best_idx, j] - pop[i, j])
                        trial[j] += np.random.normal(0, 0.1)  # Introduce adaptive weighting
                        trial[j] += np.random.uniform(-0.1, 0.1)  # Enhance search space exploration
                trial = np.clip(trial, lower_bound, upper_bound)

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial.copy()

            if evals >= self.budget:
                break

        return best