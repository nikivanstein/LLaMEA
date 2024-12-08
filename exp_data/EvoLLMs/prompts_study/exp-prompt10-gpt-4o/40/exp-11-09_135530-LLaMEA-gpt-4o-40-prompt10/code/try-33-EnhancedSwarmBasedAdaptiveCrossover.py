import numpy as np

class EnhancedSwarmBasedAdaptiveCrossover:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9
        self.w_min = 0.4
        self.alpha = 0.7  # Increased crossover rate for faster exploration
        self.evaluations = 0
        self.mutation_factor = 0.8  # Differential evolution mutation factor
    
    def initialize_particles(self):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores
    
    def differential_mutation(self, pop, i):
        indices = np.random.choice(self.population_size, 3, replace=False)
        indices = indices[indices != i]
        a, b, c = pop[indices]
        mutant = a + self.mutation_factor * (b - c)
        return np.clip(mutant, self.lb, self.ub)

    def adaptive_crossover(self, parent1, parent2):
        if np.random.rand() < self.alpha:
            cross_point = np.random.randint(1, self.dim)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        else:
            child = (parent1 + parent2) / 2
        return np.clip(child, self.lb, self.ub)
    
    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        
        while self.evaluations < self.budget:
            # Dynamically update inertia weight
            w = self.w_max - (self.w_max - self.w_min) * (self.evaluations / self.budget)
            
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
                velocities[i] = (w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.lb, self.ub)
            
            # Apply crossover and differential mutation
            if self.evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    parent1, parent2 = personal_best_positions[np.random.choice(self.population_size, 2, replace=False)]
                    child = self.adaptive_crossover(parent1, parent2)
                    mutant = self.differential_mutation(particles, i)
                    child_score = func(child)
                    mutant_score = func(mutant)
                    self.evaluations += 2
                    if child_score < personal_best_scores[i]:
                        personal_best_scores[i] = child_score
                        personal_best_positions[i] = child
                    if mutant_score < personal_best_scores[i]:
                        personal_best_scores[i] = mutant_score
                        personal_best_positions[i] = mutant
                    if child_score < global_best_score:
                        global_best_score = child_score
                        global_best_position = child
                    if mutant_score < global_best_score:
                        global_best_score = mutant_score
                        global_best_position = mutant
        
        return global_best_position, global_best_score