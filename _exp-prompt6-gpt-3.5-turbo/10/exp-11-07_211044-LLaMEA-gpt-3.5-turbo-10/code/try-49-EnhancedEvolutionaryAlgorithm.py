import numpy as np

class EnhancedEvolutionaryAlgorithm:
    def __init__(self, budget, dim, pop_size=50, num_parents=4, mutation_rate=0.1):
        self.budget, self.dim, self.pop_size, self.num_parents, self.mutation_rate = budget, dim, pop_size, num_parents, mutation_rate

    def _mutate(self, parent, step_size):
        mutation = np.random.normal(0, step_size, size=(self.dim,))
        return np.clip(parent + mutation, -5.0, 5.0)

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        for _ in range(self.budget):
            parents_idx = np.random.choice(self.pop_size, size=self.num_parents, replace=False)
            parents, step_sizes = pop[parents_idx], np.random.rand(self.num_parents) * self.mutation_rate
            mutations = np.clip(parents + np.random.normal(0, step_sizes[:, None], size=(self.num_parents, self.dim)), -5.0, 5.0)
            scores = np.array([func(ind) for ind in mutations])
            best_idx, best_parent_idx = np.argmin(scores), np.argmax([func(parent) for parent in parents])
            if scores[best_idx] < func(pop[parents_idx[best_parent_idx]]):
                pop[parents_idx[best_parent_idx]] = mutations[best_idx]
        return pop[np.argmin([func(ind) for ind in pop])]