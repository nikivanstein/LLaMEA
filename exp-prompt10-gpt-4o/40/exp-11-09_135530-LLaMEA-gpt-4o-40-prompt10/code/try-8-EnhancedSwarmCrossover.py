import numpy as np

class EnhancedSwarmCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 40
        self.c1 = 2.0  # Increased cognitive component
        self.c2 = 2.0  # Increased social component
        self.initial_w = 0.9  # Dynamic inertia weight
        self.final_w = 0.4
        self.alpha = 0.7  # Higher crossover rate
        self.evaluations = 0
        self.elite_fraction = 0.1  # Fraction of elite solutions retained
    
    def initialize_particles(self):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))  # Reduced initial velocity range
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
    
    def tournament_selection(self, scores):
        indices = np.random.choice(self.population_size, 2, replace=False)
        return indices[0] if scores[indices[0]] < scores[indices[1]] else indices[1]
    
    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        elite_count = int(self.population_size * self.elite_fraction)
        
        while self.evaluations < self.budget:
            # Dynamic inertia weight calculation
            w = self.final_w + (self.initial_w - self.final_w) * ((self.budget - self.evaluations) / self.budget)
            
            for i in range(self.population_size):
                score = func(particles[i])
                self.evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
            
            # Sort based on personal best scores and retain elites
            elite_indices = np.argsort(personal_best_scores)[:elite_count]
            elites = personal_best_positions[elite_indices]

            # Update velocities and positions
            for i in range(self.population_size):
                if i not in elite_indices:
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (w * velocities[i] +
                                     self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                     self.c2 * r2 * (global_best_position - particles[i]))
                    particles[i] = particles[i] + velocities[i]
                    # Ensure particle position within bounds
                    particles[i] = np.clip(particles[i], self.lb, self.ub)
            
            # Apply crossover using tournament selection
            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1_idx = self.tournament_selection(personal_best_scores)
                    parent2_idx = self.tournament_selection(personal_best_scores)
                    parent1, parent2 = personal_best_positions[parent1_idx], personal_best_positions[parent2_idx]
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