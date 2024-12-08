import numpy as np

class HybridGASA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.mutation_rate = 0.3
        self.cooling_rate = 0.95
        self.temperature = 1.0

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
                child = pop[i].copy()

                for j in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        child[j] += np.random.normal(0, 0.1)

                child = np.clip(child, lower_bound, upper_bound)
                f_child = func(child)
                evals += 1

                if f_child < fitness[i]:
                    pop[i] = child
                    fitness[i] = f_child
                    if f_child < fitness[best_idx]:
                        best_idx = i
                        best = child.copy()

                    if np.random.rand() < np.exp((fitness[i] - f_child) / self.temperature):
                        pop[i] = child
                        fitness[i] = f_child

            self.temperature *= self.cooling_rate

            if evals >= self.budget:
                break

        return best