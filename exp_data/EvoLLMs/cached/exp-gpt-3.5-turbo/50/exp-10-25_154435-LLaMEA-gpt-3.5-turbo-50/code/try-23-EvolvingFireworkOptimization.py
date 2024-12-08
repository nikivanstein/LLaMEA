import numpy as np

class EvolvingFireworkOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def init_sparks():
            return np.random.uniform(-5.0, 5.0, size=(self.dim,))
        
        def levy_flight(step_size):
            beta = 1.5
            sigma = (np.math.gamma(1 + beta) * np.math.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
            u = np.random.normal(0, sigma, size=(self.dim,))
            v = np.random.normal(0, 1, size=(self.dim,))
            step = u / (np.abs(v) ** (1 / beta))
            return step_size * step
        
        best_solution = init_sparks()
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            for _ in range(5):
                sparks = [init_sparks() for _ in range(10)]
                for spark in sparks:
                    step_size = np.random.uniform(0.1, 1.0)
                    new_spark = spark + levy_flight(step_size)
                    new_fitness = func(new_spark)
                    if new_fitness < best_fitness:
                        best_solution = new_spark
                        best_fitness = new_fitness

                mutant_spark = best_solution + 0.5 * (sparks[0] - sparks[1]) + 0.5 * (sparks[2] - sparks[3])
                mutant_fitness = func(mutant_spark)
                if mutant_fitness < best_fitness:
                    best_solution = mutant_spark
                    best_fitness = mutant_fitness
                    sparks[4] = mutant_spark

                step_size = np.random.uniform(0.1, 1.0)
                new_spark = best_solution + levy_flight(step_size)
                new_fitness = func(new_spark)
                if new_fitness < best_fitness:
                    best_solution = new_spark
                    best_fitness = new_fitness

        return best_solution