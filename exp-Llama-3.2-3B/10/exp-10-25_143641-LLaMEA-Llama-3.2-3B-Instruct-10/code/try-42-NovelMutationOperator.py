import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator

class NovelMutationOperator(BaseOptimizer):
    def __init__(self, budget, dim):
        super().__init__()
        self.budget = budget
        self.dim = dim
        self.stats = Stats()
        self.real = Real(['x'], [-5.0, 5.0], self.dim)
        self.refine_probability = 0.1

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.run(self.budget, func)

        # Refine the solution with a probability of 0.1
        if np.random.rand() < self.refine_probability:
            new_pop = []
            for individual in self.pop:
                if np.random.rand() < self.refine_probability:
                    new_individual = individual.copy()
                    # Randomly change one line of the individual
                    index = np.random.randint(0, self.dim)
                    new_individual[index] += np.random.uniform(-0.1, 0.1)
                    new_pop.append(new_individual)
                else:
                    new_pop.append(individual)
            self.pop = new_pop

# Usage
if __name__ == "__main__":
    optimizer = NovelMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best())}