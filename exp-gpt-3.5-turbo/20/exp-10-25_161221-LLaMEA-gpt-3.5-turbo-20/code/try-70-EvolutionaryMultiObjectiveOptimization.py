import numpy as np

class EvolutionaryMultiObjectiveOptimization:
    def __init__(self, budget, dim, objectives=2, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.objectives = objectives
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def mutate(solution):
            mutated_solution = solution + np.random.normal(0, self.mutation_rate, self.dim)
            return np.clip(mutated_solution, -5.0, 5.0)

        population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.objectives)]
        best_solution = population[0]
        best_fitness = np.mean([func(sol) for sol in population])

        for _ in range(self.budget):
            new_population = [mutate(sol) for sol in population]
            new_fitness = [func(sol) for sol in new_population]

            if np.mean(new_fitness) < best_fitness:
                best_solution = new_population[np.argmin(new_fitness)]
                best_fitness = np.mean(new_fitness)

            population = new_population

        return best_solution