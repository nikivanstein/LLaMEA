import numpy as np

class DualPhasePSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.mutation_factor = 0.9
        self.crossover_rate = 0.7
        self.c1 = 2.0
        self.c2 = 2.0
        self.max_evaluations = budget

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                # Evaluate the current particle
                score = func(self.particles[i])
                evaluations += 1
                
                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                
                if evaluations >= self.max_evaluations:
                    break

            # PSO velocity and position update
            for i in range(self.population_size):
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.particles[i])
                
                self.velocities[i] = 0.5 * self.velocities[i] + cognitive_component + social_component
                self.particles[i] = self.particles[i] + self.velocities[i]
                
                # Ensure particles remain within bounds
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
            
            # DE mutation and crossover
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant_vector = self.particles[a] + self.mutation_factor * (self.particles[b] - self.particles[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]
                
                # Evaluate trial vector
                trial_score = func(trial_vector)
                evaluations += 1
                
                # Selection
                if trial_score < self.personal_best_scores[i]:
                    self.particles[i] = trial_vector
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector
                
                if evaluations >= self.max_evaluations:
                    break
        
        return self.global_best_position, self.global_best_score