import numpy as np

class NovelAdaptiveMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.learning_rate = 0.1  # Initialize adaptive learning rate

    def __call__(self, func):
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)  # Initialize random solution
        best_fitness = func(best_solution)
        
        for eval_count in range(self.budget):
            candidate_solution = best_solution + np.random.uniform(-self.learning_rate, self.learning_rate, self.dim)  # Mutation step
            candidate_solution = np.clip(candidate_solution, -5.0, 5.0)  # Ensure solution is within bounds
            candidate_fitness = func(candidate_solution)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate_solution
                best_fitness = candidate_fitness
                self.learning_rate *= 1.1  # Increase learning rate for exploration
            else:
                mutant = np.random.uniform(-5.0, 5.0, self.dim)  # Generate mutant
                trial_solution = best_solution + 0.6 * (mutant - best_solution)  # Create trial solution
                trial_solution = np.clip(trial_solution, -5.0, 5.0)  # Ensure trial solution is within bounds
                trial_fitness = func(trial_solution)
                
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
                    self.learning_rate *= 0.9  # Decrease learning rate for exploitation
        
        return best_solution