import numpy as np

class EnhancedDEOppositionOptimization:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def opp_position(population):
            return 10.0 - population

        def evaluate(population):
            return np.array([func(ind) for ind in population])

        def bound_check(individual):
            return np.clip(individual, -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        op_population = opp_position(population)

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = bound_check(population[a] + self.f * (population[b] - population[c]))
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                op_trial = opp_position(trial)

                if func(op_trial) < func(op_population[i]):
                    new_population[i] = trial
                    op_population[i] = op_trial
                else:
                    new_population[i] = population[i]

            population = new_population

            # Adaptive parameter adjustments
            self.f = max(0.1, min(0.9, self.f + np.random.normal(0, 0.1)))
            self.cr = max(0.1, min(0.9, self.cr + np.random.normal(0, 0.1))

        best_solution = op_population[np.argmin(evaluate(op_population))]
        return best_solution