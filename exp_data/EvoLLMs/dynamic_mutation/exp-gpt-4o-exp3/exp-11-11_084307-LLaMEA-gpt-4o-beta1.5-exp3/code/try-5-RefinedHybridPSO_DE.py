import numpy as np

class RefinedHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.F_start = 0.5
        self.F_end = 0.9
        self.CR_start = 0.8
        self.CR_end = 0.95
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_evals = 0

    def __call__(self, func):
        def adapt_parameters():
            progress = self.num_evals / self.budget
            F = self.F_start + progress * (self.F_end - self.F_start)
            CR = self.CR_start + progress * (self.CR_end - self.CR_start)
            return F, CR

        def gaussian_perturbation(position, sigma=0.1):
            return np.clip(position + np.random.normal(0, sigma, self.dim), self.lower_bound, self.upper_bound)

        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        while self.num_evals < self.budget:
            F, CR = adapt_parameters()

            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                score = func(swarm[i])
                self.num_evals += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = swarm[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = swarm[i]

            # PSO update with dynamic topology
            neighborhood_size = max(1, int(self.population_size * (1 - self.num_evals / self.budget)))
            for i in range(self.population_size):
                inertia = self.w * velocities[i]
                local_best = min(personal_best_positions[np.random.choice(self.population_size, neighborhood_size)], 
                                 key=lambda pos: func(pos))
                cognitive = self.c1 * np.random.random(self.dim) * (personal_best_positions[i] - swarm[i])
                social = self.c2 * np.random.random(self.dim) * (local_best - swarm[i])
                velocities[i] = inertia + cognitive + social

                swarm[i] = swarm[i] + velocities[i]
                swarm[i] = np.clip(swarm[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if self.num_evals >= self.budget:
                    break
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(swarm[i])
                for j in range(self.dim):
                    if np.random.rand() < CR:
                        trial[j] = mutant[j]

                perturbed_trial = gaussian_perturbation(trial)
                trial_score = func(perturbed_trial)
                self.num_evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = perturbed_trial
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = perturbed_trial

        return global_best_position