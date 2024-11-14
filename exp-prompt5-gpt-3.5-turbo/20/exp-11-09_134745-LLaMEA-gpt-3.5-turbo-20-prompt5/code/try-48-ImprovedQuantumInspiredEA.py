class ImprovedQuantumInspiredEA(QuantumInspiredEA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.mutation_rates = np.full(dim, 0.2)  # Initialize mutation rates

    def apply_gate(self, individual):
        mutated_individual = individual * np.random.choice([-1, 1], size=self.dim, p=self.mutation_rates)
        self.mutation_rates = np.clip(self.mutation_rates * np.random.uniform(0.9, 1.1, size=self.dim), 0.1, 0.5)  # Update mutation rates
        return mutated_individual