# import chaospy as cp

class ImprovedEnhancedDynamicWeightQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0, mutation_rate=0.1, chaos_seed=42):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng(chaos_seed)  # Initialize random number generator for chaos map

    def __call__(self, func):
        def chaotic_map(dim):
            return cp.tent_map(self.rng.random(dim))

        particles = initialize_particles()
        best_global_position = particles[np.argmin(evaluate_fitness(particles))]
        velocities = np.zeros_like(particles)

        for _ in range(self.budget):
            for i in range(self.num_particles):
                rand1 = chaotic_map(self.dim)
                rand2 = chaotic_map(self.dim)
                adaptive_inertia = self.inertia_weight - (self.inertia_weight / self.budget) * _
                performance_ratio = func(best_global_position) / func(particles[i])
                dynamic_cognitive_weight = self.cognitive_weight * performance_ratio
                dynamic_social_weight = self.social_weight * performance_ratio

                levy_flight = np.random.standard_cauchy(self.dim) / np.sqrt(np.abs(np.random.normal(0, 1, self.dim)))
                velocities[i] = adaptive_inertia * velocities[i] + \
                                dynamic_cognitive_weight * rand1 * (best_global_position - particles[i]) + \
                                dynamic_social_weight * rand2 * (best_global_position - particles[i]) + 0.01 * levy_flight

                opposite_particle = 2 * best_global_position - particles[i]
                if func(opposite_particle) < func(particles[i]):
                    particles[i] = opposite_particle

                particles[i] = update_position(particles[i], velocities[i])

                if np.random.rand() < self.mutation_rate:
                    particles[i] += np.random.uniform(-0.5, 0.5, self.dim)

            fitness_values = evaluate_fitness(particles)
            best_particle_index = np.argmin(fitness_values)
            if fitness_values[best_particle_index] < func(best_global_position):
                best_global_position = particles[best_particle_index]

        if _ % 10 == 0 and _ != 0:
            self.num_particles = int(self.num_particles * 1.1)

        return best_global_position