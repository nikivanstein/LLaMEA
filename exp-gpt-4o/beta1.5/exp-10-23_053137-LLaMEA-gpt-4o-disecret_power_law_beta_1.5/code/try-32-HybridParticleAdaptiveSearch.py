import numpy as np

class HybridParticleAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 + 5 * self.dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
    
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = swarm.copy()
        personal_best_fitness = np.array([func(ind) for ind in swarm])
        global_best_position = personal_best_positions[np.argmin(personal_best_fitness)]
        global_best_fitness = min(personal_best_fitness)
        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update velocities and positions
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best_positions[i] - swarm[i])
                    + self.c2 * r2 * (global_best_position - swarm[i])
                )
                swarm[i] = np.clip(swarm[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Adaptive Stochastic Local Search
                if np.random.rand() < 0.25:  # 25% chance to refine the solution
                    swarm[i] = self.adaptive_stochastic_local_search(swarm[i], func, evaluations / self.budget)

                # Evaluate particle
                fitness = func(swarm[i])
                evaluations += 1

                # Update personal best
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_fitness[i] = fitness

                # Update global best
                if fitness < global_best_fitness:
                    global_best_position = swarm[i]
                    global_best_fitness = fitness

        return global_best_position

    def adaptive_stochastic_local_search(self, solution, func, progress):
        # Adaptive Stochastic Local Search: dynamically adjusts perturbation size
        step_size = 0.1 * (self.upper_bound - self.lower_bound) * (1 - progress ** 2)
        best_solution = solution.copy()
        best_fitness = func(best_solution)
        
        for _ in range(3):  # Perform a small number of local steps
            perturbation = np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(solution + perturbation, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            
            if candidate_fitness < best_fitness:
                best_solution = candidate
                best_fitness = candidate_fitness
        
        return best_solution