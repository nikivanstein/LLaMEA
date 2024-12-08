import numpy as np

class AdaptiveDynamicNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 40
        self.c1 = 1.7  # Slightly increased cognitive component
        self.c2 = 1.3  # Slightly decreased social component
        self.w_init = 0.9  # Initial inertia weight
        self.w_end = 0.4   # Final inertia weight
        self.alpha = 0.6   # Increased crossover rate
        self.evaluations = 0
        self.neighbor_count = 5  # Dynamic neighborhood size
    
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
    
    def update_inertia_weight(self):
        return self.w_end + (self.w_init - self.w_end) * (1 - self.evaluations / self.budget)
    
    def dynamic_neighborhood_best(self, scores, positions):
        neighbors = np.random.choice(self.population_size, self.neighbor_count, replace=False)
        best_neighbor = neighbors[np.argmin(scores[neighbors])]
        return positions[best_neighbor]
    
    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
            
            # Update velocities and positions with dynamic neighborhood
            inertia_weight = self.update_inertia_weight()
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                neighborhood_best = self.dynamic_neighborhood_best(personal_best_scores, personal_best_positions)
                velocities[i] = (inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (neighborhood_best - particles[i]))
                particles[i] = particles[i] + velocities[i]
                # Ensure particle position within bounds
                particles[i] = np.clip(particles[i], self.lb, self.ub)
            
            # Apply crossover
            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
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