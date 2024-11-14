import numpy as np

class EnhancedAdaptiveSwarmCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 40
        self.c1 = 2.0  # Increased cognitive coefficient
        self.c2 = 2.0  # Increased social coefficient
        self.w = 0.7   # Increased inertia weight
        self.alpha = 0.6  # Crossover rate
        self.evaluations = 0
        self.velocity_clamp = 1.5  # Velocity clamping for stability
    
    def initialize_particles(self):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores
    
    def adaptive_crossover(self, parent1, parent2):
        if np.random.rand() < self.alpha:
            cross_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        else:
            child = (parent1 + parent2) / 2
        return np.clip(child, self.lb, self.ub)
    
    def update_params(self, iteration, max_iterations):
        self.w = 0.9 - 0.5 * (iteration / max_iterations)  # Dynamic inertia
        self.c1 = 1.5 + 1.0 * (iteration / max_iterations)  # Increasing cognitive influence
        self.c2 = 2.5 - 1.0 * (iteration / max_iterations)  # Decreasing social influence
    
    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        max_iterations = self.budget // self.population_size
        
        for iteration in range(max_iterations):
            self.update_params(iteration, max_iterations)
            
            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
            
            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = np.clip(self.w * velocities[i] +
                                        self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                        self.c2 * r2 * (global_best_position - particles[i]),
                                        -self.velocity_clamp, self.velocity_clamp)
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)
            
            # Apply elitist selection and crossover
            elitist_indices = np.argsort(personal_best_scores)[:self.population_size // 2]
            for i in range(self.population_size // 2, self.population_size):
                parent1, parent2 = personal_best_positions[np.random.choice(elitist_indices, 2, replace=False)]
                child = self.adaptive_crossover(parent1, parent2)
                score = func(child)
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = child
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = child
        
        return global_best_position, global_best_score