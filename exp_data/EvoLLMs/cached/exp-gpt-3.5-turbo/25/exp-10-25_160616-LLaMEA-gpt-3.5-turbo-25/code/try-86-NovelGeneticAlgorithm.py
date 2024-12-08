import numpy as np

class NovelGeneticAlgorithm:
    def __init__(self, budget, dim, population_size=15, mutation_rate=0.1, tournament_size=3):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def mutate(solution):
            mutated_solution = np.copy(solution)
            for i in range(self.dim):
                if np.random.rand() < self.mutation_rate:
                    mutated_solution[i] = np.random.uniform(-5.0, 5.0)
            return mutated_solution

        best_solution = None
        best_fitness = np.inf

        population = initialize_population()
        for _ in range(self.budget // self.population_size):
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                tournament_indices = np.random.choice(range(self.population_size), size=self.tournament_size, replace=False)
                selected_solution = population[tournament_indices[np.argmin([evaluate_solution(population[idx]) for idx in tournament_indices])]
                mutated_solution = mutate(selected_solution)
                new_population[i] = mutated_solution

                if evaluate_solution(mutated_solution) < best_fitness:
                    best_solution = mutated_solution
                    best_fitness = evaluate_solution(mutated_solution)

            population = new_population

        return best_solution