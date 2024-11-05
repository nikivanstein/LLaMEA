import numpy as np

class Adaptive_Enhanced_Dynamic_Mutation_Population_Resizing(Enhanced_Dynamic_Mutation_Population_Resizing):
    def __init__(self, budget, dim, swarm_size=30, pso_w=0.5, pso_c1=1.5, pso_c2=1.5, de_f=0.5, de_cr=0.9, mutation_prob=0.1, mutation_scale=0.1, historical_info=[]):
        super().__init__(budget, dim, swarm_size, pso_w, pso_c1, pso_c2, de_f, de_cr, mutation_prob, mutation_scale)
        self.historical_info = historical_info
    
    def __call__(self, func):
        def enhanced_pso_de_optimizer():
            # Existing code remains unchanged up to this point
            # Integrate novel adaptive mutation strategy based on historical search information
            if self.historical_info:
                self.mutation_scale *= np.mean(self.historical_info)  # Adapt mutation scale based on historical information
                self.pso_w *= np.mean(self.historical_info)  # Adapt PSO inertia weight based on historical information

            return pso_de_optimizer()

        return enhanced_pso_de_optimizer()