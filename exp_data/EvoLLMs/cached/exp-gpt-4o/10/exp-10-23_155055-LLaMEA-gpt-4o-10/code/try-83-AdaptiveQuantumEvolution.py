import numpy as np

class AdaptiveQuantumEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.upper_bound = 5.0
        self.lower_bound = -5.0
        self.alpha = 0.9  # Quantum-inspired exploration parameter
        self.beta = 0.2  # Convergence adjustment factor
        self.mutation_factor = np.random.uniform(0.3, 0.7)  # Adaptive mutation intensity
        self.crossover_probability = 0.85  # Adjusted for strategic diversity
        self.func_evals = 0

    def quantum_position_update(self, position, best_position):
        return position + self.alpha * np.random.uniform(-1, 1, self.dim) * (best_position - position)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        # Optimization loop
        while self.func_evals < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                score = func(population[i])
                self.func_evals += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = population[i]

            # Quantum-inspired position update
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                new_position = self.quantum_position_update(population[i], global_best_position)
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                new_score = func(new_position)
                self.func_evals += 1
                if new_score < personal_best_scores[i]:
                    personal_best_scores[i] = new_score
                    personal_best_positions[i] = new_position
                    if new_score < global_best_score:
                        global_best_score = new_score
                        global_best_position = new_position

            # Differential Evolution for enhanced exploration
            if self.func_evals + self.population_size * 2 >= self.budget:
                break
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = personal_best_positions[indices]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial_vector = np.where(crossover_mask, mutant_vector, personal_best_positions[i])

                if self.func_evals < self.budget:
                    trial_score = func(trial_vector)
                    self.func_evals += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

        return global_best_position, global_best_score