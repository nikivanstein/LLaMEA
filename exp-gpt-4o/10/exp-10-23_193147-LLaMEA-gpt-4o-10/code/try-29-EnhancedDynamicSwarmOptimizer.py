import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 100
        self.inertia_weight = 0.9  # Increased initial inertia weight
        self.cognitive_weight = 1.7
        self.social_weight = 1.4
        self.mutation_prob = 0.15  # Increased mutation probability
        self.vel_bound = (self.ub - self.lb) / 5.0  # Modified velocity bound
        self.global_best_position = None
        self.global_best_value = np.inf
        self.learning_rate_decay = 0.97  # Slightly adjusted decay rate
        self.elitism_ratio = 0.1  # Increased elitism ratio
        self.mutation_scale = 0.25  # Increased mutation scale

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.vel_bound, self.vel_bound, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        
        self.global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        self.global_best_value = np.min(personal_best_values)
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            sorted_indices = np.argsort(personal_best_values)
            elitism_count = int(self.elitism_ratio * self.population_size)
            
            for i in range(self.population_size):
                if i < elitism_count:
                    particles[i] = personal_best_positions[sorted_indices[i]]
                    continue
                
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_weight * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_weight * r2 * (self.global_best_position - particles[i]))

                velocities[i] = np.clip(velocities[i], -self.vel_bound, self.vel_bound)
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)
                
                if np.random.rand() < self.mutation_prob:
                    mutation_vector = np.random.normal(0, self.mutation_scale * (self.budget - evaluations) / self.budget, self.dim)
                    particles[i] += mutation_vector
                    particles[i] = np.clip(particles[i], self.lb, self.ub)
                
                current_value = func(particles[i])
                evaluations += 1

                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]
                
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = particles[i]
                
                if evaluations >= self.budget:
                    break

            progress = evaluations / self.budget
            self.inertia_weight = 0.9 * (1 - progress) + 0.4 * progress  # Dynamic inertia weight
            self.cognitive_weight *= self.learning_rate_decay
            self.social_weight *= self.learning_rate_decay
            self.mutation_scale *= self.learning_rate_decay
        
        return self.global_best_value