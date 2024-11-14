import numpy as np

class DynamicPopSizeFireflyAlgorithmDE(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def differential_evolution(pop, f=0.5, cr=0.9):
            new_pop = np.copy(pop)
            for idx in range(len(pop)):
                a, b, c = np.random.choice(len(pop), 3, replace=False)
                r = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < cr or j == r:
                        new_pop[idx, j] = np.clip(pop[a, j] + f * (pop[b, j] - pop[c, j]), self.lb, self.ub)
            return new_pop

        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])

        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_update(pop[i]) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])

            # Differential Evolution local search
            new_pop = differential_evolution(pop)
            new_fitness = np.array([func(indiv) for indiv in new_pop])
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    pop[i] = new_pop[i]
                    fitness[i] = new_fitness[i]

            # Dynamic population size adaptation
            if np.random.rand() < 0.1:  # Probability of change
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])

        return pop[np.argmin(fitness)]