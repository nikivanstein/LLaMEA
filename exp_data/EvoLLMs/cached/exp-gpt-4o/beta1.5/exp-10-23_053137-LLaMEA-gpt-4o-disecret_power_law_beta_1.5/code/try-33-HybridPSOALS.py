import numpy as np

class HybridPSOALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + 5 * self.dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.7
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_fitness = np.array([func(ind) for ind in personal_best_positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update Velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - swarm[i]) +
                                 self.c2 * r2 * (global_best_position - swarm[i]))

                # Update Position
                swarm[i] = np.clip(swarm[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate New Position
                fitness = func(swarm[i])
                evaluations += 1

                # Update Personal Best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_fitness[i] = fitness

                # Update Global Best
                if fitness < global_best_fitness:
                    global_best_position = swarm[i]
                    global_best_fitness = fitness

                # Adaptive Stochastic Local Search
                if np.random.rand() < 0.3:  # 30% chance to refine the solution
                    swarm[i] = self.adaptive_stochastic_local_search(swarm[i], func, evaluations / self.budget)

        return global_best_position

    def adaptive_stochastic_local_search(self, solution, func, progress):
        # Adaptive Stochastic Local Search: dynamically adjusts perturbation size
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * (1 - progress)
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(5):  # Perform a small number of local steps
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution