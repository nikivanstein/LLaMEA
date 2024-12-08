import numpy as np

class DynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rate = 0.2  # Initial mutation rate

    def __call__(self, func):
        def mutation_update(pop, fitness):
            diversity = np.mean(np.std(pop, axis=0))
            mutation_factor = 1 / (1 + self.mutation_rate * diversity)
            return mutation_factor
        
        def modified_levy_update(x):
            step = levy_flight()
            new_x = x + step * np.random.normal(0, 1, self.dim) * mutation_update(pop, fitness)
            return np.clip(new_x, self.lb, self.ub)

        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])

        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = modified_levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])

            if np.random.rand() < 0.1:
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])

        return pop[np.argmin(fitness)]