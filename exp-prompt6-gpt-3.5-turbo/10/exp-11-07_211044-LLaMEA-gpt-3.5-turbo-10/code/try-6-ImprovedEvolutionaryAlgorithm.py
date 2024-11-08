import numpy as np

class ImprovedEvolutionaryAlgorithm:
    def __init__(self, budget, dim, pop_size=50, num_parents=4, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_parents = num_parents
        self.mutation_rate = mutation_rate
    
    def _mutate(self, parents, step_sizes):
        mutations = np.random.normal(0, step_sizes, size=(self.num_parents, self.dim))
        children = parents + mutations
        return np.clip(children, -5.0, 5.0)
    
    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget):
            parent_indices = np.random.choice(self.pop_size, size=self.num_parents, replace=False)
            parents = pop[parent_indices]
            step_sizes = np.random.uniform(0, self.mutation_rate, size=self.num_parents)
            
            offspring = self._mutate(parents, step_sizes)
            
            scores = np.array([func(ind) for ind in offspring])
            best_idx = np.argmin(scores)
            best_parent_idx = np.argmax([func(ind) for ind in parents])
            if scores[best_idx] < func(pop[best_parent_idx]):
                pop[best_parent_idx] = offspring[best_idx]
        
        return pop[np.argmin([func(ind) for ind in pop])]