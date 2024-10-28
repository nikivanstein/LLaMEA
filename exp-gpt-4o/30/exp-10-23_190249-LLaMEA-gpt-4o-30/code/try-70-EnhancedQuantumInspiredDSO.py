import numpy as np

class EnhancedQuantumInspiredDSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 40
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.cognitive = 1.5
        self.social = 2.0
        self.quantum_prob = 0.3
        self.elite_quantum_prob = 0.2
        self.global_best_position = None
        self.global_best_value = np.inf
        self.max_velocity = (self.ub - self.lb) * 0.2
        self.de_mutation_factor = 0.8
        self.de_crossover_rate = 0.9

    def __call__(self, func):
        np.random.seed(42)
        positions = self.halton_sequence(self.num_particles, self.dim)
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.num_particles, np.inf)

        for i in range(self.num_particles):
            value = func(positions[i])
            personal_best_values[i] = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best_position = np.copy(positions[i])

        eval_count = self.num_particles

        while eval_count < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (eval_count / self.budget))
            for i in range(self.num_particles):
                if np.random.rand() < self.quantum_prob:
                    memory_factor = 0.6
                    quantum_position = self.global_best_position + memory_factor * np.random.randn(self.dim)
                    quantum_position = np.clip(quantum_position, self.lb, self.ub)
                    quantum_value = func(quantum_position)
                    eval_count += 1
                    if quantum_value < personal_best_values[i]:
                        personal_best_values[i] = quantum_value
                        personal_best_positions[i] = quantum_position

                if np.random.rand() < self.elite_quantum_prob:
                    elite_quantum_position = personal_best_positions[i] + np.random.randn(self.dim)
                    elite_quantum_position = np.clip(elite_quantum_position, self.lb, self.ub)
                    elite_quantum_value = func(elite_quantum_position)
                    eval_count += 1
                    if elite_quantum_value < personal_best_values[i]:
                        personal_best_positions[i] = elite_quantum_position
                        personal_best_values[i] = elite_quantum_value

                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    inertia * velocities[i] +
                    self.cognitive * r1 * (personal_best_positions[i] - positions[i]) +
                    self.social * r2 * (self.global_best_position - positions[i])
                )
                velocities[i] = np.clip(velocities[i], -self.max_velocity, self.max_velocity)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)

                value = func(positions[i])
                eval_count += 1
                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = np.copy(positions[i])

                if np.random.rand() < self.de_crossover_rate:
                    idxs = np.random.choice(self.num_particles, 3, replace=False)
                    donor_vector = (
                        positions[idxs[0]] +
                        self.de_mutation_factor * (positions[idxs[1]] - positions[idxs[2]])
                    )
                    trial_vector = np.where(np.random.rand(self.dim) < self.de_crossover_rate,
                                            donor_vector, positions[i])
                    trial_vector = np.clip(trial_vector, self.lb, self.ub)

                    trial_value = func(trial_vector)
                    eval_count += 1
                    if trial_value < personal_best_values[i]:
                        positions[i] = trial_vector
                        personal_best_positions[i] = trial_vector
                        personal_best_values[i] = trial_value

                if eval_count >= self.budget:
                    break

            if eval_count >= self.budget:
                break

            for i in range(self.num_particles):
                value = personal_best_values[i]
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = np.copy(personal_best_positions[i])

        return self.global_best_position, self.global_best_value

    def halton_sequence(self, size, dim):
        def halton_single_index(index, base):
            result = 0.0
            f = 1.0
            i = index
            while i > 0:
                f = f / base
                result = result + f * (i % base)
                i = int(i / base)
            return result

        seq = np.empty((size, dim))
        primes = self._first_primes(dim)
        for d in range(dim):
            for i in range(size):
                seq[i, d] = halton_single_index(i + 1, primes[d]) * (self.ub - self.lb) + self.lb
        return seq

    def _first_primes(self, n):
        primes = []
        candidate = 2
        while len(primes) < n:
            is_prime = True
            for p in primes:
                if candidate % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(candidate)
            candidate += 1
        return primes