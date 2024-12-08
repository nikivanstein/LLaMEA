import numpy as np

class EnhancedDynamicInertiaWeightEnhancedSocialSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 40
        self.phi_p = 1.5
        self.phi_g = 2.0

    def __call__(self, func):
        def _initialize_swarm():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def _local_search(x, f, step_size):
            for _ in range(40):  
                x_new = x + step_size * np.random.normal(size=self.dim)
                if f(x_new) < f(x):
                    x = x_new
                    step_size *= 1.15  
                else:
                    step_size *= 0.85  
            return x

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
                mutation_scale = 0.1 + 0.9 * np.random.uniform() * (1 - _ / self.budget) 
                inertia_weight = 0.4 + 0.4 * ((self.budget - _) / self.budget)  
                local_fitness_landscape = np.max([func(swarm[j]) for j in range(self.swarm_size) if j != i])
                step_size = 0.1 + 0.4 * np.exp(-2 * local_fitness_landscape)  
                swarm[i] = best_position + mutation_scale * (swarm[i] - best_position)  
                swarm[i] = _local_search(swarm[i], func, step_size)
                best_position = swarm[i] if func(swarm[i]) < func(best_position) else best_position
        return best_position