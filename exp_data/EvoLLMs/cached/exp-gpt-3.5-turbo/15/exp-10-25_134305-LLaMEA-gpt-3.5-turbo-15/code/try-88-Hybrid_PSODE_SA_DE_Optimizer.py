import numpy as np
from scipy.optimize import differential_evolution

class Hybrid_PSODE_SA_DE_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4, initial_temperature=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def __call__(self, func):
        def hybrid_pso_sa_de_helper():
            # PSO initialization
            self.inertia_weight = np.random.uniform(0.4, 0.9)
            # PSO optimization loop
            # Simulated Annealing (SA) initialization
            current_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
            current_cost = func(current_solution)
            best_solution = current_solution
            best_cost = current_cost
            # SA optimization loop
            for _ in range(int(self.budget*0.5)):  # Spend 50% of the budget on SA
                proposal_solution = current_solution + np.random.normal(0, 0.1, size=self.dim)
                proposal_solution = np.clip(proposal_solution, -5.0, 5.0)
                proposal_cost = func(proposal_solution)
                if proposal_cost < current_cost or np.random.rand() < np.exp((current_cost - proposal_cost) / self.initial_temperature):
                    current_solution = proposal_solution
                    current_cost = proposal_cost
                    if proposal_cost < best_cost:
                        best_solution = proposal_solution
                        best_cost = proposal_cost
            # Differential Evolution (DE) optimization
            de_result = differential_evolution(func, bounds=[(-5.0, 5.0)]*self.dim, maxiter=int(self.budget*0.5))  # Spend 50% of the budget on DE
            if de_result.fun < best_cost:
                best_solution = de_result.x
            return best_solution

        return hybrid_pso_sa_de_helper()

optimizer = Hybrid_PSODE_SA_DE_Optimizer(budget=1000, dim=10)