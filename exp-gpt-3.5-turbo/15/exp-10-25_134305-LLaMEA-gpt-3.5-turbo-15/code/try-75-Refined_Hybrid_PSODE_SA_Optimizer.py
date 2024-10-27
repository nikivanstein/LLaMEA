import numpy as np
from scipy.optimize import differential_evolution

class Refined_Hybrid_PSODE_SA_Optimizer(Hybrid_PSODE_SA_Optimizer):
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4, initial_temperature=100, cooling_rate=0.95):
        super().__init__(budget, dim, num_particles, max_iterations, inertia_weight, cognitive_weight, social_weight, initial_temperature, cooling_rate)

    def __call__(self, func):
        def refined_hybrid_pso_de_sa_helper():
            # Refined PSO initialization
            self.inertia_weight = np.random.uniform(0.5, 0.8)  # Refined inertia weight update
            # Refined SA optimization loop
            for _ in range(int(self.budget*0.65)):  # Refined 65% budget allocation for SA
                proposal_solution = current_solution + np.random.normal(0, 0.15, size=self.dim)  # Refined SA proposal generation
                proposal_solution = np.clip(proposal_solution, -4.5, 4.5)  # Refined solution clipping
                proposal_cost = func(proposal_solution)
            # Refined DE optimization
            de_result = differential_evolution(func, bounds=[(-4.5, 4.5)]*self.dim, maxiter=int(self.budget*0.35))  # Refined 35% budget allocation for DE
            return de_result.x if de_result.fun < best_cost else best_solution

        return refined_hybrid_pso_de_sa_helper()