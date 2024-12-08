import numpy as np

class MemoryEnhancedOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increase population size for diversified search
        self.inertia_weight = 0.4  # Adjusted inertia weight for balanced exploration-exploitation
        self.cognitive_constant = 1.4  # Slightly reduced cognitive constant to avoid premature convergence
        self.social_constant = 1.8  # Enhanced social constant to leverage collective intelligence
        self.mutation_factor = 0.8  # Reduced mutation factor for stability
        self.crossover_rate = 0.8  # Tweaked crossover rate to maintain diversity
        self.elitism_rate = 0.15  # Increased elitism to ensure progress retention
        self.memory_factor = 0.1  # Introduced memory factor to retain useful historical solutions

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))  # Narrower velocity range
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf
        historical_best_positions = []

        evaluations = 0

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += self.population_size

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]
                    historical_best_positions.append(global_best_position)

            sorted_indices = np.argsort(scores)
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = sorted_indices[:elite_count]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.copy(particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial_vector[j] = mutant_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    particles[i] = trial_vector
                    scores[i] = trial_score

            if evaluations + 1 >= self.budget:
                break

            for i in range(elite_count):
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[elite_indices[i]] + np.random.uniform(-0.1, 0.1, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[elite_indices[i]]:
                    particles[elite_indices[i]] = local_candidate
                    scores[elite_indices[i]] = local_score

            if evaluations + 1 >= self.budget:
                break

            if historical_best_positions and len(historical_best_positions) > 10:
                historical_best_positions = historical_best_positions[-10:]
                for mem_pos in historical_best_positions:
                    mem_candidate = mem_pos + self.memory_factor * np.random.uniform(-1.0, 1.0, self.dim)
                    mem_candidate = np.clip(mem_candidate, self.lower_bound, self.upper_bound)
                    mem_score = func(mem_candidate)
                    evaluations += 1
                    if mem_score < global_best_score:
                        global_best_score = mem_score
                        global_best_position = mem_candidate

        return global_best_position, global_best_score