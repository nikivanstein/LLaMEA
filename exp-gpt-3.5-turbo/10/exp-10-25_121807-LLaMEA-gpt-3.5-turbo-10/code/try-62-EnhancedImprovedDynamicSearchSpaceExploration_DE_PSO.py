import numpy as np

class EnhancedImprovedDynamicSearchSpaceExploration_DE_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        best_solution = np.random.uniform(lower_bound, upper_bound, size=self.dim)
        best_fitness = func(best_solution)
        step_size = 0.1 * (upper_bound - lower_bound)  # Adaptive step size
        for _ in range(self.budget):
            # Introduce Combined Levy Flights and Particle Swarm Optimization for exploring new solutions
            levy_step = np.random.standard_cauchy(size=self.dim) / np.sqrt(np.abs(np.random.normal(size=self.dim)))  
            new_solution_levy = best_solution + levy_step * step_size
            new_solution_levy = np.clip(new_solution_levy, lower_bound, upper_bound)
            new_fitness_levy = func(new_solution_levy)
            if new_fitness_levy < best_fitness:
                best_solution = new_solution_levy
                best_fitness = new_fitness_levy
                step_size *= 0.95  # Self-adaptive strategy enhancement
                
            # Particle Swarm Optimization strategy to further improve exploitation
            w = 0.5  # Inertia weight
            c1 = 1.5  # Cognitive weight
            c2 = 1.5  # Social weight
            velocity = np.random.uniform(-1, 1, size=self.dim)
            personal_best = best_solution.copy()
            global_best = best_solution.copy()
            for i in range(self.dim):
                velocity[i] = w * velocity[i] + c1 * np.random.rand() * (personal_best[i] - best_solution[i]) + c2 * np.random.rand() * (global_best[i] - best_solution[i])
                best_solution[i] += velocity[i]
            best_solution = np.clip(best_solution, lower_bound, upper_bound)
            new_fitness_pso = func(best_solution)
            if new_fitness_pso < best_fitness:
                best_fitness = new_fitness_pso
                step_size *= 0.95  # Self-adaptive strategy enhancement
        
        return best_solution