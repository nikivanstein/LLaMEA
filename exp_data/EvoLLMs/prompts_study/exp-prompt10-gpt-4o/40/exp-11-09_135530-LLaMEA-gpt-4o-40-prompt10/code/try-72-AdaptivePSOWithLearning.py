import numpy as np

class AdaptivePSOWithLearning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.alpha = 0.7
        self.beta = 0.9
        self.gamma = 0.1  # Learning rate
        self.evaluations = 0
    
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
    
    def differential_mutation(self, target, best, r1, r2):
        mutated = target + self.beta * (best - target) + self.beta * (r1 - r2)
        return np.clip(mutated, self.lb, self.ub)
    
    def apply_learning(self, particles, personal_best_positions, global_best_position):
        for i in range(self.population_size):
            knowledge_gain = self.gamma * (global_best_position - particles[i])
            particles[i] = particles[i] + knowledge_gain
            particles[i] = np.clip(particles[i], self.lb, self.ub)
    
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

            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
            
            for i in range(self.population_size):
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                r1, r2 = personal_best_positions[r1], personal_best_positions[r2]
                velocities[i] = (w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = self.differential_mutation(particles[i], global_best_position, r1, r2)
                particles[i] = np.clip(particles[i], self.lb, self.ub)
                
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
            
            self.apply_learning(particles, personal_best_positions, global_best_position)
        
        return global_best_position, global_best_score