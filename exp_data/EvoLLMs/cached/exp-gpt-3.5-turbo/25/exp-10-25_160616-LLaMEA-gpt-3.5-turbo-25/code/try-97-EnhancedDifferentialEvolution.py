import numpy as np

class EnhancedDifferentialEvolution(DifferentialEvolution):
    def __init__(self, budget, dim, population_size=10, scaling_factor=0.8, crossover_rate=0.7, stages=3):
        super().__init__(budget, dim, population_size, scaling_factor, crossover_rate)
        self.stages = stages

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
                
                trial_solutions = [np.where(np.random.uniform(0, 1, self.dim) < self.crossover_rate, donor_solution, target_solution) for _ in range(self.stages)]
                trial_fitnesses = [evaluate_solution(trial_sol) for trial_sol in trial_solutions]
                
                best_trial_index = np.argmin(trial_fitnesses)
                best_trial_solution = trial_solutions[best_trial_index]
                
                if trial_fitnesses[best_trial_index] < evaluate_solution(target_solution):
                    population[i] = best_trial_solution

                if trial_fitnesses[best_trial_index] < best_fitness:
                    best_solution = best_trial_solution
                    best_fitness = trial_fitnesses[best_trial_index]

        return best_solution