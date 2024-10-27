# import numpy as np

class Adaptive_Mutation_Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 20
        self.max_iter = 100
        self.mutation_factor = np.full(self.num_particles, 0.5)

    def __call__(self, func):
        def generate_initial_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.num_particles, self.dim))

        def pso_update(particles, velocities, p_best, g_best):
            inertia_weight = 0.5
            c1 = 1.5
            c2 = 1.5

            for i in range(self.num_particles):
                r1, r2 = np.random.random(), np.random.random()
                velocities[i] = inertia_weight * velocities[i] + c1 * r1 * (p_best[i] - particles[i]) + c2 * r2 * (g_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

        def de_mutation(particles, best_solution):
            for i in range(self.num_particles):
                idxs = np.arange(self.num_particles)
                np.random.shuffle(idxs)
                r1, r2, r3 = particles[idxs[0]], particles[idxs[1]], particles[idxs[2]]
                mutant = particles[i] + self.mutation_factor[i] * (r1 - r2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                if func(mutant) < func(particles[i]):
                    particles[i] = mutant
                    self.mutation_factor[i] *= 1.1  # Adaptive mutation adjustment
                else:
                    self.mutation_factor[i] *= 0.9

        def simulated_annealing(current_solution, current_value, best_solution, best_value, temperature, cooling_rate):
            new_solution = current_solution + np.random.normal(0, 0.1, self.dim)
            new_solution = np.clip(new_solution, self.lower_bound, self.upper_bound)
            new_value = func(new_solution)

            if new_value < current_value or np.random.rand() < np.exp((current_value - new_value) / temperature):
                current_solution, current_value = new_solution, new_value

            if new_value < best_value:
                best_solution, best_value = new_solution, new_value

            temperature *= cooling_rate

            return current_solution, current_value, best_solution, best_value, temperature

        particles = generate_initial_population()
        velocities = np.zeros_like(particles)
        p_best = particles.copy()
        g_best = particles[np.argmin([func(p) for p in particles])]
        best_solution = g_best
        best_value = func(g_best)
        temperature = 1.0
        cooling_rate = 0.99

        for _ in range(self.max_iter):
            pso_update(particles, velocities, p_best, g_best)
            de_mutation(particles, best_solution)
            for i in range(self.num_particles):
                particles[i], _, p_best[i], _, temperature = simulated_annealing(particles[i], func(particles[i]), p_best[i], func(p_best[i]), temperature, cooling_rate)
            
            g_best = particles[np.argmin([func(p) for p in particles])]
            if func(g_best) < best_value:
                best_solution, best_value = g_best, func(g_best)

        return best_solution