import numpy as np

class Cooperative_Bat_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

        # Bat Algorithm Parameters
        self.num_bats = 40  # Number of bats (equivalent to particles)
        self.loudness = 0.5  # Loudness parameter
        self.pulse_rate = 0.5  # Pulse emission rate
        self.frequency_min = 0.0
        self.frequency_max = 1.0

        # Adaptive Differential Evolution Parameters
        self.F = 0.7  # Scaling factor
        self.CR = 0.9  # Crossover probability

        # Initialize positions, velocities, and best solutions
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_bats, self.dim))
        self.velocities = np.zeros((self.num_bats, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_bats, np.inf)

        # Global best
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        evals = 0

        while evals < self.budget:
            # Evaluate each bat
            scores = np.apply_along_axis(func, 1, self.positions)
            evals += self.num_bats

            # Update personal and global bests
            for i in range(self.num_bats):
                if scores[i] < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = scores[i]
                    self.personal_best_positions[i] = self.positions[i]

                if scores[i] < self.global_best_score:
                    self.global_best_score = scores[i]
                    self.global_best_position = self.positions[i]

            # Update velocities and positions (Bat Algorithm)
            for i in range(self.num_bats):
                freq = self.frequency_min + (self.frequency_max - self.frequency_min) * np.random.rand()
                self.velocities[i] += (self.positions[i] - self.global_best_position) * freq
                new_position = self.positions[i] + self.velocities[i]

                if np.random.rand() > self.pulse_rate:
                    new_position = self.global_best_position + self.loudness * np.random.randn(self.dim)

                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                new_score = func(new_position)
                evals += 1

                # Accept the new solutions based on the loudness and update
                if new_score < scores[i] and np.random.rand() < self.loudness:
                    self.positions[i] = new_position
                    scores[i] = new_score

            # Adaptive Differential Evolution
            for i in range(self.num_bats):
                idx1, idx2, idx3 = np.random.choice(range(self.num_bats), 3, replace=False)
                x1, x2, x3 = self.positions[idx1], self.positions[idx2], self.positions[idx3]
                mutant_vector = np.clip(x1 + self.F * (x2 - x3), self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.positions[i])
                
                trial_score = func(trial_vector)
                evals += 1

                # DE acceptance criterion
                if trial_score < scores[i]:
                    self.positions[i] = trial_vector
                    scores[i] = trial_score

        return self.global_best_position, self.global_best_score