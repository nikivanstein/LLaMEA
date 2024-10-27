import numpy as np
import random
import copy

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            new_individual = self.evaluate_fitness(func)
            if np.isnan(new_individual) or np.isinf(new_individual):
                raise ValueError("Invalid function value")
            if new_individual < 0 or new_individual > 1:
                raise ValueError("Function value must be between 0 and 1")
            if self.func_evaluations >= self.budget:
                break

            # Select parent
            parent1, parent2 = random.sample(self.population, 2)
            parent1 = copy.deepcopy(parent1)
            parent2 = copy.deepcopy(parent2)

            # Crossover
            child = (parent1 + parent2) / 2

            # Mutation
            if random.random() < self.mutation_rate:
                child[0] += np.random.uniform(-1, 1)
                child[1] += np.random.uniform(-1, 1)

            # Replace with new individual
            self.population[0] = child

    def evaluate_fitness(self, func):
        individual = func(self.search_space)
        return individual

class HEBBOEvolutionaryStrategy(HEBBO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        # Select parent using tournament selection
        tournament_size = 3
        tournament_results = []
        for _ in range(tournament_size):
            tournament_results.append(self.evaluate_fitness(func))
        tournament_results = np.array(tournament_results)
        tournament_indices = np.argsort(tournament_results)[:self.population_size // 2]
        parent1 = func(self.search_space[tournament_indices])
        parent2 = func(self.search_space[tournament_indices[::-1]])
        parent1 = copy.deepcopy(parent1)
        parent2 = copy.deepcopy(parent2)

        # Crossover
        child = (parent1 + parent2) / 2

        # Mutation
        if random.random() < self.mutation_rate:
            child[0] += np.random.uniform(-1, 1)
            child[1] += np.random.uniform(-1, 1)

        # Replace with new individual
        self.population[0] = child

class HEBBOEvolutionaryStrategyWithAdaptation(HEBBOEvolutionaryStrategy):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.adaptation_rate = 0.1

    def __call__(self, func):
        # Select parent using tournament selection
        tournament_size = 3
        tournament_results = []
        for _ in range(tournament_size):
            tournament_results.append(self.evaluate_fitness(func))
        tournament_results = np.array(tournament_results)
        tournament_indices = np.argsort(tournament_results)[:self.population_size // 2]
        parent1 = func(self.search_space[tournament_indices])
        parent2 = func(self.search_space[tournament_indices[::-1]])
        parent1 = copy.deepcopy(parent1)
        parent2 = copy.deepcopy(parent2)

        # Crossover
        child = (parent1 + parent2) / 2

        # Mutation
        if random.random() < self.mutation_rate:
            child[0] += np.random.uniform(-1, 1)
            child[1] += np.random.uniform(-1, 1)

        # Replace with new individual
        self.population[0] = child

# Example usage:
if __name__ == "__main__":
    func = lambda x: np.sin(x)
    hebbo = HEBBO(100, 10)
    hebboEvolutionaryStrategy = HEBBOEvolutionaryStrategy(100, 10)
    hebboEvolutionaryStrategyEvolutionaryStrategy = HEBBOEvolutionaryStrategyWithAdaptation(100, 10)
    hebboEvolutionaryStrategyEvolutionaryStrategy.__call__(func)