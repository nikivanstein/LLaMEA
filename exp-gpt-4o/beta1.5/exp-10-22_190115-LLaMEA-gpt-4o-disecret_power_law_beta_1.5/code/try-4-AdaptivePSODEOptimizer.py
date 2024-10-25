import numpy as np

class AdaptivePSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50  # Size of the population
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = (-1.0, 1.0)
        self.c1 = 1.49445  # Adjusted cognitive coefficient
        self.c2 = 1.49445  # Adjusted social coefficient
        self.w = 0.729  # Adjusted inertia weight using constriction factor
        self.de_F = 0.9  # Slightly increased DE scaling factor
        self.de_CR = 0.8  # Reduced DE crossover probability for more exploration

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([np.inf] * self.pop_size)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0

        # Main loop
        while evaluations < self.budget:
            # Evaluate current particles
            scores = np.array([func(p) for p in particles])
            evaluations += self.pop_size

            # Update personal and global bests
            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]
            
            # PSO Update
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            cognitive_component = self.c1 * r1 * (personal_best_positions - particles)
            social_component = self.c2 * r2 * (global_best_position - particles)
            velocities = self.w * velocities + cognitive_component + social_component
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)
            
            # DE Update
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(particles[a] + self.de_F * (particles[b] - particles[c]), self.lower_bound, self.upper_bound)
                trial = np.copy(particles[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.de_CR or j == j_rand:
                        trial[j] = mutant[j]
                if evaluations < self.budget and func(trial) < scores[i]:
                    particles[i] = trial
                    scores[i] = func(trial)
                    evaluations += 1

        return global_best_position, global_best_score