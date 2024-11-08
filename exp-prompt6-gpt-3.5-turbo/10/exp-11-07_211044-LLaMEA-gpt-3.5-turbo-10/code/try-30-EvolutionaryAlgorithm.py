import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim, pop_size=50, num_parents=4, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.num_parents = num_parents
        self.mutation_rate = mutation_rate
    
    def _mutate(self, parent, step_size):
        mutation = np.random.normal(0, step_size, size=self.dim)
        child = parent + mutation
        return np.clip(child, -5.0, 5.0)
    
    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        
        for _ in range(self.budget):
            parents = pop[np.random.choice(self.pop_size, size=self.num_parents, replace=False)]
            offspring = []
            for parent in parents:
                step_size = np.random.uniform(0, self.mutation_rate)
                child = self._mutate(parent, step_size)
                offspring.append(child)
            offspring = np.array(offspring)
            
            scores = np.array([func(ind) for ind in offspring])
            best_idx = np.argmin(scores)
            if scores[best_idx] < func(pop[np.argmax(scores)]):
                pop[np.argmax(scores)] = offspring[best_idx]
        
        return pop[np.argmin([func(ind) for ind in pop])]