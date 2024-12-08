import numpy as np
from scipy.optimize import differential_evolution

class Novel_Metaheuristic_Optimizer:
    def __init__(self, budget, dim, num_particles=30, max_iterations=1000, inertia_weight=0.7, cognitive_weight=1.4, social_weight=1.4, initial_temperature=100, cooling_rate=0.95):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def novel_metaheuristic_helper():
            # PSO initialization
            # Dynamic adjustment of inertia weight based on exploration-exploitation balance
            inertia_weight = np.random.uniform(0.4, 0.9)  # Update the inertia weight dynamically
            # PSO optimization loop
            # Simulated Annealing (SA) initialization
            current_solution = np.random.uniform(-5.0, 5.0, size=self.dim)
            current_cost = func(current_solution)
            best_solution = current_solution
            best_cost = current_cost
            # SA optimization loop
            for _ in range(int(self.budget*0.7)):  # Spend 70% of the budget on SA
                proposal_solution = current_solution + np.random.normal(0, 0.1, size=self.dim)
                proposal_solution = np.clip(proposal_solution, -5.0, 5.0)
                proposal_cost = func(proposal_solution)
                if proposal_cost < current_cost or np.random.rand() < np.exp((current_cost - proposal_cost) / 100):
                    current_solution = proposal_solution
                    current_cost = proposal_cost
                    if proposal_cost < best_cost:
                        best_solution = proposal_solution
                        best_cost = proposal_cost
            # Differential Evolution (DE) optimization
            de_result = differential_evolution(func, bounds=[(-5.0, 5.0)]*self.dim, maxiter=int(self.budget*0.3))  # Spend 30% of the budget on DE
            if de_result.fun < best_cost:
                best_solution = de_result.x
            return best_solution

        return novel_metaheuristic_helper()