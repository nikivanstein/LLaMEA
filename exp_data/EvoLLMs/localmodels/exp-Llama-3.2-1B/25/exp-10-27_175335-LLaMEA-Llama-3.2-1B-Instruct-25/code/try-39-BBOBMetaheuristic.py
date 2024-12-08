import numpy as np

class BBOBMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        if self.func_evals >= self.budget:
            raise ValueError("Not enough evaluations left to optimize the function")

        func_evals = self.func_evals
        self.func_evals += 1
        return func

    def search(self, func):
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol = None
        for _ in range(10):
            sol = np.random.uniform(bounds, size=self.dim)
            func_sol = self.__call__(func, sol)
            if func_sol < self.__call__(func, sol):
                sol = sol
        return sol

class EvolutionaryBBOBMetaheuristic(BBOBMetaheuristic):
    def __init__(self, budget, dim, mutation_rate):
        super().__init__(budget, dim)
        self.mutation_rate = mutation_rate

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(0, self.dim)
            self.func_evals -= 1
            self.func_evals += 1
            individual[idx] = np.random.uniform(bounds[0], bounds[1])
            return individual
        return individual

    def __call__(self, func):
        bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
        sol = self.search(func)
        if self.func_evals < self.budget:
            new_individual = self.mutate(sol)
            func_evals = self.func_evals
            self.func_evals += 1
            return new_individual
        else:
            return sol

# Description: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# Code: 
# ```python
# BBOBMetaheuristic: Evolutionary Algorithm for Black Box Optimization using Genetic Programming
# 
# class EvolutionaryBBOBMetaheuristic(BBOBMetaheuristic):
#     def __init__(self, budget, dim, mutation_rate):
#         super().__init__(budget, dim)
#         self.mutation_rate = mutation_rate

#     def mutate(self, individual):
#         if np.random.rand() < self.mutation_rate:
#             idx = np.random.randint(0, self.dim)
#             self.func_evals -= 1
#             self.func_evals += 1
#             individual[idx] = np.random.uniform(bounds[0], bounds[1])
#             return individual
#         return individual

#     def __call__(self, func):
#         bounds = np.linspace(-5.0, 5.0, self.dim, endpoint=False)
#         sol = self.search(func)
#         if self.func_evals < self.budget:
#             new_individual = self.mutate(sol)
#             func_evals = self.func_evals
#             self.func_evals += 1
#             return new_individual
#         else:
#             return sol