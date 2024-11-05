import numpy as np

class SwarmDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.9
        self.differential_weight = 0.8
        self.inertia_weight = 0.5
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                       size=(self.population_size, self.dim))
        velocities = np.random.uniform(low=-0.1, high=0.1, size=(self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        eval_count = self.population_size
        
        while eval_count < self.budget:
            # Particle Swarm Optimization update
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best_positions[i] - population[i])
                social_velocity = self.social_coeff * r2 * (global_best_position - population[i])
                # Stochastic adjustment of inertia weight
                dynamic_inertia = self.inertia_weight * np.random.uniform(0.9, 1.1)
                velocities[i] = (dynamic_inertia * velocities[i] +
                                 cognitive_velocity + social_velocity)
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

            # Evaluate new positions
            scores = np.array([func(ind) for ind in population])
            eval_count += self.population_size

            # Update personal and global bests
            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = population[i]
            global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_idx] < global_best_score:
                global_best_position = personal_best_positions[global_best_idx]
                global_best_score = personal_best_scores[global_best_idx]

            # Differential Evolution update
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