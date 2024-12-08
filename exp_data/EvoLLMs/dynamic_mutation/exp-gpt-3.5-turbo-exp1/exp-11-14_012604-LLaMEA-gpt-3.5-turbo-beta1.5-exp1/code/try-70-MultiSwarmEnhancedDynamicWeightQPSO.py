# import numpy as np

class MultiSwarmEnhancedDynamicWeightQPSO:
    def __init__(self, budget, dim, num_particles=30, num_swarms=3, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.num_swarms = num_swarms
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_swarms, self.num_particles, self.dim))

        def update_position(particles, velocities):
            return np.clip(particles + velocities, -5.0, 5.0)

        def evaluate_fitness(particles):
            return np.array([[func(p) for p in swarm] for swarm in particles])

        particles = initialize_particles()
        best_global_positions = particles[np.argmin(evaluate_fitness(particles))]
        velocities = np.zeros_like(particles)

        for _ in range(self.budget):
            for s in range(self.num_swarms):
                for i in range(self.num_particles):
                    rand1 = np.random.rand(self.dim)
                    rand2 = np.random.rand(self.dim)
                    adaptive_inertia = self.inertia_weight - (self.inertia_weight / self.budget) * _
                    performance_ratio = func(best_global_positions[s]) / func(particles[s][i])
                    dynamic_cognitive_weight = self.cognitive_weight * performance_ratio
                    dynamic_social_weight = self.social_weight * performance_ratio

                    levy_flight = np.random.standard_cauchy(self.dim) / np.sqrt(np.abs(np.random.normal(0, 1, self.dim)))
                    velocities[s][i] = adaptive_inertia * velocities[s][i] + \
                                      dynamic_cognitive_weight * rand1 * (best_global_positions[s] - particles[s][i]) + \
                                      dynamic_social_weight * rand2 * (best_global_positions[s] - particles[s][i]) + 0.01 * levy_flight

                    opposite_particle = 2 * best_global_positions[s] - particles[s][i]
                    if func(opposite_particle) < func(particles[s][i]):
                        particles[s][i] = opposite_particle

                    particles[s][i] = update_position(particles[s][i], velocities[s][i])

                    if np.random.rand() < self.mutation_rate:
                        particles[s][i] += np.random.uniform(-0.5, 0.5, self.dim)

                swarm_fitness = evaluate_fitness(particles[s])
                best_particle_index = np.argmin(swarm_fitness)
                if swarm_fitness[best_particle_index] < func(best_global_positions[s]):
                    best_global_positions[s] = particles[s][best_particle_index]

            if _ % 10 == 0 and _ != 0:
                self.num_particles = int(self.num_particles * 1.1)  # Increase particle number dynamically every 10 iterations

        return best_global_positions[np.argmin(evaluate_fitness(particles))]