import numpy as np

class EnhancedMetaheuristicAlgorithmLevyADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_scale = 0.5  # Initialize mutation scale

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            if eval_count % (self.budget // 10) == 0 and eval_count > 0:
                self.mutation_scale = 0.5 - 0.5 * eval_count / self.budget  # Adapt mutation scale

            if np.random.rand() < 0.1:  # Introduce Levy flight with a probability
                levy_step = 0.01 * np.random.standard_cauchy(self.dim)  # Generate Levy flight step
                candidate_solution = best_solution + levy_step
            else:
                if np.random.rand() < 0.5:  # Introduce adaptive differential evolution
                    mutant = np.random.uniform(-5.0, 5.0, self.dim)  # Generate mutant for DE
                    trial_solution = best_solution + 0.6 * (mutant - best_solution)  # Create trial solution
                else:
                    candidate_solution = best_solution + np.random.uniform(-self.mutation_scale, self.mutation_scale, self.dim)  # Mutation step

            if 'candidate_solution' in locals():  # Check if candidate solution exists
                candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
                candidate_fitness = func(candidate_solution)
                
                if candidate_fitness < best_fitness:
                    best_solution = candidate_solution
                    best_fitness = candidate_fitness
                else:
                    mutant = np.random.uniform(-5.0, 5.0, self.dim)  # Generate mutant
                    trial_solution = best_solution + 0.6 * (mutant - best_solution)  # Create trial solution
                    trial_solution = np.clip(trial_solution, -5.0, 5.0)  # Ensure trial solution is within bounds
                    trial_fitness = func(trial_solution)
                    
                    if trial_fitness < best_fitness:
                        best_solution = trial_solution
                        best_fitness = trial_fitness
        
        return best_solution