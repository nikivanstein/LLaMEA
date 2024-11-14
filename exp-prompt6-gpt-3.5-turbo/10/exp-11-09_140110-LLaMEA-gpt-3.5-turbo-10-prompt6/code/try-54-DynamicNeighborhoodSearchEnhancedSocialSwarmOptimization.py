import numpy as np

class DynamicNeighborhoodSearchEnhancedSocialSwarmOptimization(DynamicInertiaWeightEnhancedSocialSwarmOptimization):
    def __call__(self, func):
        def _neighborhood_search(x, f):
            neighborhoods = [x + np.random.normal(size=self.dim) for _ in range(5)]  # Dynamic neighborhood exploration
            neighborhood_fitness = [f(neighbor) for neighbor in neighborhoods]
            best_neighbor = neighborhoods[np.argmin(neighborhood_fitness)]
            return best_neighbor if f(best_neighbor) < f(x) else x

        swarm = _initialize_swarm()
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
                swarm[i] = _neighborhood_search(swarm[i], func)  # Neighborhood search strategy
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position