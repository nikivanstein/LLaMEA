import numpy as np

class SwarmDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.9
        self.differential_weight = 0.8
        self.inertia_weight = 0.9
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_bound = 0.1

    def __call__(self, func):
        population = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                       size=(self.population_size, self.dim))
        velocities = np.random.uniform(low=-self.velocity_bound, high=self.velocity_bound, size=(self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        eval_count = self.population_size
        
        while eval_count < self.budget:
            self.inertia_weight = 0.9 - 0.5 * (eval_count / self.budget)
            improvement = np.mean(personal_best_scores) - global_best_score
            dynamic_velocity_bound = 0.1 * (1.0 + improvement / np.abs(self.lower_bound))
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions[i] - population[i])
                social_velocity = self.social_coeff * r2 * (global_best_position - population[i])
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 cognitive_velocity + social_velocity)
                velocities[i] = np.clip(velocities[i], -dynamic_velocity_bound, dynamic_velocity_bound)  # Adaptive clip velocity
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in population])
            eval_count += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]
            global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_idx] < global_best_score:
                global_best_position = personal_best_positions[global_best_idx]
                global_best_score = personal_best_scores[global_best_idx]

            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_weight = self.differential_weight + 0.2 * (np.random.rand() - 0.5)
                mutant_vector = population[a] + adaptive_weight * (population[b] - population[c])
                trial_vector = np.copy(population[i])
                adaptive_crossover = self.crossover_rate * (0.5 + 0.5 * np.sin(np.pi * eval_count / self.budget))
                for j in range(self.dim):
                    if np.random.rand() < adaptive_crossover:
                        trial_vector[j] = mutant_vector[j]
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                trial_score = func(trial_vector)
                eval_count += 1
                if trial_score < scores[i]:
                    new_population[i] = trial_vector
                    scores[i] = trial_score
            population = new_population

        return global_best_position, global_best_score