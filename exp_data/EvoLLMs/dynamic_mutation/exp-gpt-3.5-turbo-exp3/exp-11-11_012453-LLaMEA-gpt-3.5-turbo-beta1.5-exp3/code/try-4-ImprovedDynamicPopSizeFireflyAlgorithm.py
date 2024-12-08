import numpy as np

class ImprovedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.crossover_prob = 0.5  # Crossover probability
        self.scale_factor = 0.5  # Scale factor for differential evolution
        
    def __call__(self, func):
        def differential_evolution(pop, fitness):
            new_pop = np.copy(pop)
            for i in range(self.pop_size):
                a, b, c = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant = pop[a] + self.scale_factor * (pop[b] - pop[c])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, pop[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
            return new_pop, fitness

        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])
        
        for _ in range(self.budget):
            pop, fitness = differential_evolution(pop, fitness)
            
            # Dynamic population size adaptation
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])
        
        return pop[np.argmin(fitness)]