import numpy as np

class EnhancedPSODEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = (-1.0, 1.0)
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.5
        self.de_F = 0.8
        self.de_CR = 0.9
        self.adaptive_rate = 0.1

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([np.inf] * self.pop_size)
        global_best_position = None
        global_best_score = np.inf
        
        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(p) for p in particles])
            evaluations += self.pop_size

            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]
            
            self.c1 = max(0.4, self.c1 - self.adaptive_rate * np.random.rand())
            self.c2 = min(2.2, self.c2 + self.adaptive_rate * np.random.rand())
            self.w = max(0.1, self.w - self.adaptive_rate * np.random.rand())

            r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
            for i in range(self.pop_size):
                cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social_component = self.c2 * r2 * (global_best_position - particles[i])
                velocity_decay = self.w * velocities[i] * r3
                velocities[i] = velocity_decay + cognitive_component + social_component
                particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)
            
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
                trial_score = func(trial)
                if evaluations < self.budget and trial_score < scores[i]:
                    particles[i] = trial
                    scores[i] = trial_score
                    evaluations += 1

        return global_best_position, global_best_score