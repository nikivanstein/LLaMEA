import numpy as np

class AdaptiveHybridPSOGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        
        self.population_size = 50  # Increased population for better search space coverage
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        
        self.pbest_positions = np.copy(self.particles)
        self.pbest_scores = np.full(self.population_size, np.inf)
        
        self.gbest_position = None
        self.gbest_score = np.inf
        
        # Dynamic learning rates
        self.c1_initial, self.c1_final = 2.0, 0.3
        self.c2_initial, self.c2_final = 0.3, 2.0
        self.w_initial, self.w_final = 0.8, 0.3

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Calculate adaptive parameters
            progress = evaluations / self.budget
            self.c1 = self.c1_initial - progress * (self.c1_initial - self.c1_final)
            self.c2 = self.c2_initial + progress * (self.c2_final - self.c2_initial)
            self.w = self.w_initial - progress * (self.w_initial - self.w_final)
            
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1

                if score < self.pbest_scores[i]:
                    self.pbest_scores[i] = score
                    self.pbest_positions[i] = self.particles[i]

                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = self.particles[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions for adaptive PSO with inertia
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                social = self.c2 * r2 * (self.gbest_position - self.particles[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if evaluations < self.budget:
                # Genetic Algorithm crossover and mutation
                elite_count = self.population_size // 5  # Elitism strategy
                elite_indices = np.argsort(self.pbest_scores)[:elite_count]
                elite_population = self.particles[elite_indices]

                for i in range(self.population_size):
                    if i not in elite_indices:
                        parent1, parent2 = np.random.choice(elite_population, 2, replace=False)
                        cross_point = np.random.randint(1, self.dim)
                        trial = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                        mutation = np.random.normal(0, 0.1, self.dim)
                        trial += mutation
                        trial = np.clip(trial, self.lower_bound, self.upper_bound)

                        trial_score = func(trial)
                        evaluations += 1

                        if trial_score < self.pbest_scores[i]:
                            self.pbest_scores[i] = trial_score
                            self.pbest_positions[i] = trial

                        if trial_score < self.gbest_score:
                            self.gbest_score = trial_score
                            self.gbest_position = trial

                        if evaluations >= self.budget:
                            break

        return self.gbest_score, self.gbest_position