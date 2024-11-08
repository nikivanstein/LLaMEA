import numpy as np
from joblib import Parallel, delayed

class Parallel_PSO_SA_Optimizer:
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

            best_solution = None
            best_fitness = float('inf')

            particles = np.random.uniform(bounds[0], bounds[1], (n_particles, self.dim))
            velocities = np.zeros((n_particles, self.dim))

            for _ in range(max_iterations_pso):
                results = Parallel(n_jobs=-1)(delayed(update_particle)(i, particles, velocities, bounds, alpha, best_solution) for i in range(n_particles))
                particles, velocities, best_solution, best_fitness = zip(*results)

            current_solution = best_solution[0]
            current_fitness = best_fitness[0]
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

        def update_particle(i, particles, velocities, bounds, alpha, best_solution):
            fitness = objective_function(particles[i])
            if fitness < best_fitness:
                best_solution = particles[i].copy()
                best_fitness = fitness

            cognitive_component = np.random.random() * (particles[i] - particles[i])
            social_component = np.random.random() * (best_solution - particles[i])
            velocities[i] = alpha * (velocities[i] + cognitive_component + social_component)
            particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])

            return particles, velocities, best_solution, best_fitness

        return pso_sa_optimization()