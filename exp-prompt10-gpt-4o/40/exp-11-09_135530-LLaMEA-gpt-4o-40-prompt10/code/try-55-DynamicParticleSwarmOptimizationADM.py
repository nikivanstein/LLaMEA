import numpy as np

class DynamicParticleSwarmOptimizationADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w_max = 0.9  # Dynamic adjustment of inertia weight
        self.w_min = 0.4
        self.F = 0.5  # Differential mutation factor
        self.CR = 0.9  # Crossover probability for differential mutation
        self.evaluations = 0
    
    def initialize_particles(self):
        particles = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        return particles, velocities, personal_best_positions, personal_best_scores
    
    def differential_mutation(self, pop, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(pop[a] + self.F * (pop[b] - pop[c]), self.lb, self.ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial
    
    def __call__(self, func):
        particles, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_position = None
        global_best_score = np.inf
        
        while self.evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))
            
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
            
            # Apply differential mutation and crossover
            if self.evaluations + self.population_size <= self.budget:
                new_population = np.copy(particles)
                for i in range(self.population_size):
                    mutant = self.differential_mutation(particles, i)
                    trial = self.crossover(particles[i], mutant)
                    score = func(trial)
                    self.evaluations += 1
                    if score < personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = trial
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = trial
                    new_population[i] = trial if score < func(particles[i]) else particles[i]
                particles = new_population
        
        return global_best_position, global_best_score