import numpy as np

class ImprovedDynamicPopSizeFireflyAlgorithm(DynamicPopSizeFireflyAlgorithm):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.beta0 = 1.0  # Initial value for beta
        self.gamma = 1.0  # Initial value for gamma

    def __call__(self, func):
        def levy_flight(x, beta):
            sigma1 = (np.prod(np.power(np.arange(1, self.dim+1), -beta)))**(1/self.dim)
            sigma2 = np.power(np.random.standard_normal(self.dim) * sigma1, 1/beta)
            return x + sigma2 * np.random.normal(0, 1, self.dim)

        def adjust_beta_gamma():
            self.beta0 *= 0.95
            self.gamma *= 1.05

        pop = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(indiv) for indiv in pop])

        for _ in range(self.budget):
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]:
                        step_size = attraction(pop[i], pop[j])
                        pop[i] = levy_flight(pop[i], self.beta0) if step_size > np.random.rand() else pop[i]
                        fitness[i] = func(pop[i])

            if np.random.rand() < 0.1:
                self.pop_size = min(30, self.pop_size + 5)
                pop = np.vstack((pop, np.random.uniform(self.lb, self.ub, (5, self.dim)))
                fitness = np.append(fitness, [func(indiv) for indiv in pop[-5:]])

            adjust_beta_gamma()

        return pop[np.argmin(fitness)]