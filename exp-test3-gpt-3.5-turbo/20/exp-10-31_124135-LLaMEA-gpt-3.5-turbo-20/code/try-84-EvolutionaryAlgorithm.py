import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def tournament_selection(self, fitness, tournament_size):
        selected_indices = np.random.choice(range(len(fitness)), tournament_size, replace=False)
        tournament_contestants = [fitness[i] for i in selected_indices]
        return selected_indices[np.argmin(tournament_contestants)]

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        tournament_size = min(5, self.budget)  # Define tournament size

        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = self.tournament_selection(fitness, tournament_size)  # Use tournament selection
            best_solution = population[best_idx]

            # Mutation operator based on Gaussian distribution
            mutated_population = population + np.random.normal(0, 0.1, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget
        
        return best_solution