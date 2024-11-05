import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        tournament_size = 3  # Define tournament size

        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]

            # Tournament selection for parent selection
            selected_parents = []
            for _ in range(self.budget):
                tournament_indices = np.random.choice(self.budget, size=tournament_size, replace=False)
                tournament_fitness = [fitness[i] for i in tournament_indices]
                selected_parents.append(population[tournament_indices[np.argmin(tournament_fitness)]])

            # Mutation operator based on Gaussian distribution
            mutated_population = np.array(selected_parents) + np.random.normal(0, 0.1, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget

        return best_solution