import numpy as np

class EvoHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.hm_size = 5
        self.hm_accept_rate = 0.1
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
                hm = pop[np.argsort(fitness)[:self.hm_size]]
                
                for j in range(self.dim):
                    if np.random.rand() < self.hm_accept_rate:
                        trial[j] = hm[np.random.randint(0, self.hm_size), j]
                    else:
                        trial[j] += np.random.normal(0, 0.1)
                
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