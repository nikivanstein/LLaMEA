import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w = 0.7   # inertia weight
        self.F = 0.5   # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        # Initialize particles for PSO
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)

        # Initialize population for DE
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        scores = np.full(self.population_size, np.inf)

        # Evaluate initial particles and population
        for i in range(self.population_size):
            score = func(positions[i])
            personal_best_scores[i] = score
            scores[i] = score

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # PSO Update
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.c2 * r2 * (global_best_position - positions[i]))
                new_position = positions[i] + velocities[i]
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_score = func(new_position)
                evaluations += 1
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = new_position

                positions[i] = new_position

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(population[a] + self.F * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == np.random.randint(self.dim):
                        trial[j] = mutant[j]

                # Evaluate trial vector
                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    population[i] = trial

            # Update Global Best
            global_best_idx = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_idx]
            global_best_score = personal_best_scores[global_best_idx]

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_score