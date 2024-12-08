import numpy as np

class EnhancedPSOWithAdaptiveNeighborhood:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.diversity_threshold = 0.5
        self.global_learning_rate = 0.1

    def __call__(self, func):
        np.random.seed(0)

        position = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.population_size, np.inf)

        global_best_value = np.inf
        global_best_position = np.zeros(self.dim)

        eval_count = 0

        def clip_to_bounds(particles):
            return np.clip(particles, self.bounds[0], self.bounds[1])

        def calculate_entropy(population):
            hist, _ = np.histogramdd(population, bins=10, range=[self.bounds]*self.dim)
            prob = hist / np.sum(hist)
            prob = prob[prob > 0]
            return -np.sum(prob * np.log2(prob))

        while eval_count < self.budget:
            for i in range(self.population_size):
                current_value = func(position[i])
                eval_count += 1
                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i]
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i]

            entropy = calculate_entropy(position)
            self.inertia_weight = 0.9 if entropy < self.diversity_threshold else 0.6

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (self.inertia_weight * velocity +
                        self.cognitive_coeff * r1 * (personal_best_position - position) +
                        self.social_coeff * r2 * (global_best_position - position))
            position = clip_to_bounds(position + velocity)

            # Introduce adaptive neighborhood mutation
            for i in range(self.population_size):
                neighbors = np.random.choice(self.population_size, 5, replace=False)
                best_in_neighborhood = min(neighbors, key=lambda x: personal_best_value[x])
                mutant_vector = position[i] + self.mutation_factor * (position[best_in_neighborhood] - position[i])
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, position[i])
                trial_vector = clip_to_bounds(trial_vector)

                trial_value = func(trial_vector)
                eval_count += 1
                if trial_value < personal_best_value[i]:
                    personal_best_value[i] = trial_value
                    personal_best_position[i] = trial_vector
                if trial_value < global_best_value:
                    global_best_value = trial_value
                    global_best_position = trial_vector

                position[i] = position[i] + self.global_learning_rate * (global_best_position - position[i])
                position[i] = clip_to_bounds(position[i])

                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_value