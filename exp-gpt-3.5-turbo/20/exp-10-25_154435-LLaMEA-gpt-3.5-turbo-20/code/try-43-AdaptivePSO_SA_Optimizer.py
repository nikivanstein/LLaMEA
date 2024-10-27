import numpy as np

class AdaptivePSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 20
        self.max_iter = 100

    def __call__(self, func):
        def generate_initial_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_particles, self.dim))

        def adaptive_pso_update(particles, velocities, p_best, g_best, inertia_weights):
            for i in range(self.num_particles):
                r1, r2 = np.random.random(), np.random.random()
                inertia_weight = inertia_weights[i]
                velocities[i] = inertia_weight * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

        inertia_weights = np.ones(self.num_particles) * 0.5
        c1 = 1.5
        c2 = 1.5

        particles = generate_initial_population()
        velocities = np.zeros_like(particles)
        p_best = particles.copy()
        g_best = particles[np.argmin([func(p) for p in particles])]
        best_solution = g_best
        best_value = func(g_best)
        temperature = 1.0
        cooling_rate = 0.99

        for _ in range(self.max_iter):
            adaptive_pso_update(particles, velocities, p_best, g_best, inertia_weights)
            for i in range(self.num_particles):
                particles[i], _, p_best[i], _, temperature = simulated_annealing(particles[i], func(particles[i]), p_best[i], func(p_best[i]), temperature, cooling_rate)
            
            g_best = particles[np.argmin([func(p) for p in particles])]
            if func(g_best) < best_value:
                best_solution, best_value = g_best, func(g_best)

            # Adaptive update of inertia weights based on individual particle performance
            for i in range(self.num_particles):
                if func(particles[i]) < func(p_best[i]):
                    inertia_weights[i] *= 1.1
                else:
                    inertia_weights[i] *= 0.9

        return best_solution