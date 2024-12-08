import numpy as np

class ImprovedEnhancedDynamicWeightQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0, mutation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        def initialize_particles():
            return np.random.uniform(-5.0, 5.0, size=(self.num_particles, self.dim))

        def update_position(particles, velocities):
            return np.clip(particles + velocities, -5.0, 5.0)

        def evaluate_fitness(particles):
            return np.array([func(p) for p in particles])

        particles = initialize_particles()
        best_global_position = particles[np.argmin(evaluate_fitness(particles))]
        velocities = np.zeros_like(particles)

        for _ in range(self.budget):
            for i in range(self.num_particles):
                rand1 = np.random.rand(self.dim)
                rand2 = np.random.rand(self.dim)
                adaptive_inertia = self.inertia_weight - (self.inertia_weight / self.budget) * _
                performance_ratio = func(best_global_position) / func(particles[i])
                dynamic_cognitive_weight = self.cognitive_weight * performance_ratio
                dynamic_social_weight = self.social_weight * performance_ratio

                # Introduce Levy flight behavior
                levy_flight = np.random.standard_cauchy(self.dim) / np.sqrt(np.abs(np.random.normal(0, 1, self.dim)))
                velocities[i] = adaptive_inertia * velocities[i] + \
                                dynamic_cognitive_weight * rand1 * (best_global_position - particles[i]) + \
                                dynamic_social_weight * rand2 * (best_global_position - particles[i]) + 0.01 * levy_flight
                                
                # Introduce opposition-based learning for better exploration
                opposite_particle = 2 * best_global_position - particles[i]
                if func(opposite_particle) < func(particles[i]):
                    particles[i] = opposite_particle

                particles[i] = update_position(particles[i], velocities[i])

                # Mutation operator
                if np.random.rand() < self.mutation_rate:
                    particles[i] += np.random.uniform(-0.5, 0.5, self.dim)

            fitness_values = evaluate_fitness(particles)
            best_particle_index = np.argmin(fitness_values)
            if fitness_values[best_particle_index] < func(best_global_position):
                best_global_position = particles[best_particle_index]
        
        if _ % 10 == 0 and _ != 0:
            if _ % 100 == 0:  # Dynamic population resizing every 100 iterations
                self.num_particles = np.clip(int(self.num_particles * 1.1), 1, 100)  # Restrict particle number within a range

        return best_global_position