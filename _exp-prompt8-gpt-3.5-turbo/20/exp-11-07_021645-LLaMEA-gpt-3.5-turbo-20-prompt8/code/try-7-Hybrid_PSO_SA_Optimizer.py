import numpy as np

class Hybrid_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def hybrid_optimization():
            def objective_function(x):
                return func(x)
            
            n_particles = 10
            max_iterations_total = 110  # Combined max iterations
            max_iterations_sa = 10
            bounds = (-5.0, 5.0)
            alpha = 0.95
            T0 = 1.0
            T_min = 0.0001
            sigma = 0.1

            best_solution = None
            best_fitness = float('inf')

            particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))

            current_solution = particles[0]  # Initialize current solution

            for iteration in range(1, max_iterations_total + 1):
                T = T0 * (T_min / T0) ** (iteration / max_iterations_total)  # Update temperature

                for i in range(n_particles):
                    for _ in range(max_iterations_sa):
                        candidate_solution = np.clip(current_solution + np.random.normal(0, sigma, self.dim), bounds[0], bounds[1])
                        candidate_fitness = objective_function(candidate_solution)

                        if candidate_fitness < objective_function(current_solution) or np.random.rand() < np.exp((objective_function(current_solution) - candidate_fitness) / T):
                            current_solution = candidate_solution

                for i in range(n_particles):
                    fitness = objective_function(particles[i])
                    if fitness < best_fitness:
                        best_solution = particles[i].copy()
                        best_fitness = fitness

                    cognitive_component = np.random.random() * (particles[i] - particles[i])
                    social_component = np.random.random() * (best_solution - particles[i])
                    velocities[i] = alpha * (velocities[i] + cognitive_component + social_component)
                    particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

            return best_solution

        return hybrid_optimization()