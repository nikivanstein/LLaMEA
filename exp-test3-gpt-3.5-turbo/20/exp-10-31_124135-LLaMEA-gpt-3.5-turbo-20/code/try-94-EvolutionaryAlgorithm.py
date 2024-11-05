import numpy as np

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def tournament_selection(self, fitness, tournament_size):
        selected_indices = []
        for _ in range(self.budget):
            tournament_indices = np.random.choice(len(fitness), tournament_size, replace=False)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            selected_indices.append(tournament_indices[np.argmin(tournament_fitness)])
        return selected_indices

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = 0
        tournament_size = int(0.1 * self.budget)  # Define tournament size as 10% of the population size
        while evals < self.budget:
            fitness = [func(ind) for ind in population]
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Tournament selection for diversity
            selected_indices = self.tournament_selection(fitness, tournament_size)
            selected_population = population[selected_indices]
            
            # Mutation operator based on Gaussian distribution
            mutated_population = selected_population + np.random.normal(0, 0.1, (self.budget, self.dim))
            population = mutated_population
            evals += self.budget
        
        return best_solution