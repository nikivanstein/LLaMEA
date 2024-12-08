import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, population_size=10, scaling_factor=0.8, crossover_rate=0.7):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.scaling_factor = scaling_factor
        self.crossover_rate = crossover_rate

    def __call__(self, func):
        def evaluate_solution(solution):
            return func(solution)

        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        best_solution = None
        best_fitness = np.inf

        population = initialize_population()
        for _ in range(self.budget // self.population_size):
            for i in range(self.population_size):
                target_solution = population[i]
                indices = np.arange(self.population_size)
                np.random.shuffle(indices)
                base_solution = population[indices[0]]
                donor_solution = base_solution + self.scaling_factor * (population[indices[1]] - population[indices[2]])
                trial_solution = np.where(np.random.uniform(0, 1, self.dim) < self.crossover_rate, donor_solution, target_solution)
                
                target_fitness = evaluate_solution(target_solution)
                trial_fitness = evaluate_solution(trial_solution)
                if trial_fitness < target_fitness:
                    population[i] = trial_solution

                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness

        return best_solution