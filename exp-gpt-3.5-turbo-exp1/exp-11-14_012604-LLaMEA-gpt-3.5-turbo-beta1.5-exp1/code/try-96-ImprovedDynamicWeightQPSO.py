import numpy as np

class ImprovedDynamicWeightQPSO:
    def __init__(self, budget, dim, num_particles=30, inertia_weight=0.9, cognitive_weight=2.0, social_weight=2.0, mutation_rate=0.1, de_weight=0.5, de_cross_prob=0.9, de_adaptive_prob=0.5):  # Change: Added de_adaptive_prob
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.mutation_rate = mutation_rate
        self.de_weight = de_weight
        self.de_cross_prob = de_cross_prob
        self.de_adaptive_prob = de_adaptive_prob  # Change: Added de_adaptive_prob

    def __call__(self, func):
        def initialize_particles():
            chaotic_map = lambda x: 4 * x * (1 - x)
            initial_positions = np.zeros((self.num_particles, self.dim))
            for i in range(self.num_particles):
                initial_positions[i] = chaotic_map(np.random.rand(self.dim))
            return initial_positions

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
                                
                # Introduce differential evolution with adaptive crossover probability
                r1, r2, r3 = np.random.choice(range(self.num_particles), 3, replace=False)
                diff_vector = particles[r2] - particles[r3]
                mutated_vector = particles[i] + self.de_weight * diff_vector
                crossover_mask = np.random.rand(self.dim) < (self.de_cross_prob * (1 - _ / self.budget) + self.de_adaptive_prob * (_ / self.budget))  # Change: Adaptive crossover probability
                trial_vector = np.where(crossover_mask, mutated_vector, particles[i])
                if func(trial_vector) < func(particles[i]):
                    particles[i] = trial_vector

                particles[i] = update_position(particles[i], velocities[i])

                # Mutation operator with dynamic mutation rate adjustment
                improvement_ratio = func(particles[i]) / func(best_global_position)
                self.mutation_rate = max(0.1, min(self.mutation_rate * (1 + improvement_ratio), 0.9)) if improvement_ratio > 1 else self.mutation_rate
                if np.random.rand() < self.mutation_rate:
                    particles[i] += np.random.uniform(-0.5, 0.5, self.dim)

            fitness_values = evaluate_fitness(particles)
            best_particle_index = np.argmin(fitness_values)
            if fitness_values[best_particle_index] < func(best_global_position):
                best_global_position = particles[best_particle_index]
        
        if _ % 10 == 0 and _ != 0:
            self.num_particles = int(self.num_particles * 1.1)  # Change: Increase particle number dynamically every 10 iterations

        return best_global_position