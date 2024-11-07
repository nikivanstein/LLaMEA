import numpy as np

class EnhancedMetaheuristicAlgorithmLevyADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.5  # Initialize mutation scale
        self.population_size = 10  # Initialize population size

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale
                self.population_size = 10 + int(40 * eval_count / self.budget)  # Adapt population size

            if np.random.rand() < 0.1:  # Introduce Levy flight with a probability
                levy_step = 0.01 * np.random.standard_cauchy(self.dim)  # Generate Levy flight step
                candidate_solution = best_solution + levy_step
            else:
                if np.random.rand() < 0.5 and self.population_size > 1:  # Introduce adaptive differential evolution
                    mutants = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))  # Generate mutants for DE
                    trial_solutions = best_solution + 0.6 * (mutants - best_solution)  # Create trial solutions
                    trial_solutions = np.clip(trial_solutions, -5.0, 5.0)  # Ensure solutions are within bounds
                    candidate_fitnesses = np.apply_along_axis(func, 1, trial_solutions)
                    best_idx = np.argmin(candidate_fitnesses)
                    candidate_solution = trial_solutions[best_idx]
                else:
                    candidate_solution = best_solution + np.random.uniform(-self.mutation_scale, self.mutation_scale, self.dim)  # Mutation step

            if 'candidate_solution' in locals():  # Check if candidate solution exists
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
                candidate_fitness = func(candidate_solution)
                
                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
        
        return best_solution