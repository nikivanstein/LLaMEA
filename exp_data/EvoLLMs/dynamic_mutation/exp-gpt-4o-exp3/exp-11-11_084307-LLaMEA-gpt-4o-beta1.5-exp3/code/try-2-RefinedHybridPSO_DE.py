import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.49445  # Cognitive component
        self.c2 = 1.49445  # Social component
        self.w_max = 0.9   # Max inertia weight
        self.w_min = 0.4   # Min inertia weight
        self.F = 0.8       # DE mutation factor
        self.CR_max = 0.9  # Max DE crossover probability
        self.CR_min = 0.1  # Min DE crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0

    def __call__(self, func):
        # Initialize swarm
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        while self.num_evals < self.budget:
            # Calculate adaptive parameters
            w = self.w_max - ((self.w_max - self.w_min) * (self.num_evals / self.budget))
            CR = self.CR_max - ((self.CR_max - self.CR_min) * (self.num_evals / self.budget))

            # Evaluate current swarm
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                score = func(swarm[i])
                self.num_evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # PSO update
            for i in range(self.population_size):
                inertia = w * velocities[i]
                cognitive = self.c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = self.c2 * np.random.random(self.dim) * (global_best_position - swarm[i])
                velocities[i] = inertia + cognitive + social

                # Update position
                swarm[i] = swarm[i] + velocities[i]
                # Ensure bounds
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            # DE-inspired mutation and crossover with adaptive CR
            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                self.num_evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial

        return global_best_position