import numpy as np

class ImprovedDynamicInertiaWeightEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def _dynamic_mutation(self, x, f):
        mutation_rate = 0.2 + 0.8 * np.random.uniform()  # Dynamic mutation rate adaptation
        for _ in range(self.dim):
            if np.random.uniform() < mutation_rate:
                x[_] = np.clip(x[_] + np.random.normal(), -5.0, 5.0)  # Mutation with normal distribution
        return x

    def __call__(self, func):
        swarm = self._initialize_swarm()
        best_position = swarm[np.argmin([func(x) for x in swarm])]
        for _ in range(self.budget):
            diversity = np.mean(np.std(swarm, axis=0))
            new_swarm_size = max(5, int(self.swarm_size * (1 - diversity / 10.0)))
            selected_indices = np.random.choice(range(self.swarm_size), new_swarm_size, replace=False)
            swarm = swarm[selected_indices]
            self.swarm_size = new_swarm_size
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1)
                inertia_weight = 0.5 + 0.5 * np.random.uniform()
                swarm[i] = inertia_weight * swarm[i] + self.phi_p * r_p * (best_position - swarm[i]) + self.phi_g * r_g * (best_position - swarm[i])
                swarm[i] = self._local_search(swarm[i], func)
                swarm[i] = self._dynamic_mutation(swarm[i], func)  # Integrate dynamic mutation operation
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position