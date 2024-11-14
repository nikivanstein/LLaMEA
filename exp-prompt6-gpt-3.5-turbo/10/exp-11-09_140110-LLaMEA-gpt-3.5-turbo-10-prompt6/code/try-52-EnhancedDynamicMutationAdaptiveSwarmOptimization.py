import numpy as np

class EnhancedDynamicMutationAdaptiveSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _local_search(x, f):
            step_size = 1.0
            for _ in range(10):
                x_new = x + step_size * np.random.normal(size=self.dim)
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.1  # Adaptive step adjustment
                else:
                    step_size *= 0.9  # Adaptive step adjustment
            return x

        swarm = _initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))  # Dynamic population size adjustment
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                inertia_weight = 0.5 + 0.5 * np.random.uniform()  # Dynamic inertia weight adaptation
                mutation_rate = 0.1 * (1 - func(swarm[i]) / func(best_position))  # Dynamic mutation strategy
                swarm[i] = inertia_weight * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                swarm[i] = swarm[i] + mutation_rate * np.random.normal(size=self.dim)  # Mutation based on individual performance
                swarm[i] = _local_search(swarm[i], func)
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position