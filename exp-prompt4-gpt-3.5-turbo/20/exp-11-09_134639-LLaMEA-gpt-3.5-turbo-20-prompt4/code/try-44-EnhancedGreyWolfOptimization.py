import numpy as np

class EnhancedGreyWolfOptimization(GreyWolfOptimization):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        def dynamic_search_space():
            lb = self.lb - np.sqrt(np.mean((population - np.mean(population, axis=0)) ** 2, axis=0))
            ub = self.ub + np.sqrt(np.mean((population - np.mean(population, axis=0)) ** 2, axis=0))
            return lb, ub

        population = initialize_population()
        fitness = [func(individual) for individual in population]
        best_index = np.argmin(fitness)
        best_solution = population[best_index]

        for _ in range(self.budget - self.budget):
            a = 2 - 2 * _ / self.budget
            lb, ub = dynamic_search_space()

            for i in range(self.budget):
                A = 2 * a * np.random.rand(self.dim) - a
                C = 2 * np.random.rand(self.dim)
                P = np.random.rand(self.dim)

                if i < self.budget / 2:
                    D = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D
                    population[i] = np.clip(X1, lb, ub)
                else:
                    D1 = np.abs(C * best_solution - population[i])
                    X1 = best_solution - A * D1
                    population[i] = np.clip(X1 + levy_flight(), lb, ub)

            fitness = [func(individual) for individual in population]
            new_best_index = np.argmin(fitness)
            if fitness[new_best_index] < fitness[best_index]:
                best_index = new_best_index
                best_solution = population[best_index]

        return best_solution