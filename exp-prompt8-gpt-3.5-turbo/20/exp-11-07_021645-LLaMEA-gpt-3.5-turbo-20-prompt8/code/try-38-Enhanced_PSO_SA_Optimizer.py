import numpy as np

class Enhanced_PSO_SA_Optimizer:
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
            inertia_max = 0.9
            inertia_min = 0.4

            best_solution = None
            best_fitness = float('inf')

            particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))
            inertia_weight = inertia_max

            random_cognitive = np.random.random((n_particles, self.dim))
            random_social = np.random.random((n_particles, self.dim))

            for _ in range(max_iterations_pso):
                new_velocities = inertia_weight * velocities + alpha * (random_cognitive * (particles - particles) + random_social * (best_solution - particles))
                particles = np.clip(particles + new_velocities, bounds[0], bounds[1])

                current_fitness_values = list(map(objective_function, particles))
                best_particle_index = np.argmin(current_fitness_values)
                if current_fitness_values[best_particle_index] < best_fitness:
                    best_solution = particles[best_particle_index].copy()
                    best_fitness = current_fitness_values[best_particle_index]

                inertia_weight = max(inertia_max - (_ / max_iterations_pso) * (inertia_max - inertia_min), inertia_min)

            current_solution = best_solution
            current_fitness = best_fitness
            T = T0

            while T > T_min:
                for _ in range(max_iterations_sa):
                    candidate_solution = np.clip(current_solution + np.random.normal(0, sigma, self.dim), bounds[0], bounds[1])
                    candidate_fitness = objective_function(candidate_solution)

                    if candidate_fitness < current_fitness or np.random.rand() < np.exp((current_fitness - candidate_fitness) / T):
                        current_solution = candidate_solution
                        current_fitness = candidate_fitness

                T *= alpha

            return current_solution

        return pso_sa_optimization()
