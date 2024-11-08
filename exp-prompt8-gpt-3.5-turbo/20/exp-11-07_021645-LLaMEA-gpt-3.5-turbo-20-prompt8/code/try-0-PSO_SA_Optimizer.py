import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def pso_sa_optimization():
            def objective_function(x):
                return func(x)
            
            n_particles = 10
            max_iterations_pso = 100
            max_iterations_sa = 10
            bounds = (-5.0, 5.0)
            alpha = 0.95
            T0 = 1.0
            T_min = 0.0001
            sigma = 0.1

            def within_bounds(x):
                return np.clip(x, bounds[0], bounds[1])

            best_solution = None
            best_fitness = float('inf')

            # Initialize particles
            particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))

            for _ in range(max_iterations_pso):
                for i in range(n_particles):
                    fitness = objective_function(particles[i])
                    if fitness < best_fitness:
                        best_solution = particles[i]
                        best_fitness = fitness

                    # Update particle velocity
                    cognitive_component = np.random.random() * (particles[i] - particles[i])
                    social_component = np.random.random() * (best_solution - particles[i])
                    velocities[i] = alpha * (velocities[i] + cognitive_component + social_component)

                    # Update particle position
                    particles[i] = within_bounds(particles[i] + velocities[i])

            # Simulated Annealing
            current_solution = best_solution
            current_fitness = best_fitness
            T = T0

            while T > T_min:
                for _ in range(max_iterations_sa):
                    candidate_solution = within_bounds(current_solution + np.random.normal(0, sigma, self.dim))
                    candidate_fitness = objective_function(candidate_solution)

                    if candidate_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - candidate_fitness) / T):
                        current_solution = candidate_solution
                        current_fitness = candidate_fitness

                T *= alpha

            return current_solution

        return pso_sa_optimization()