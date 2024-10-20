import numpy as np
import matplotlib.pyplot as plt


def discrete_power_law_distribution(n, beta):
    def discrete_power_law(n, alpha, beta):
        half_n = int(n/2)
        C_beta_half_n = 0
        for i in range(1, half_n+1):
            C_beta_half_n += i**(-beta)
        probability_alpha = C_beta_half_n**(-1) * alpha**(-beta)
        return probability_alpha
    half_n = int(n/2)
    elements = [alpha for alpha in range(1, half_n+1)]
    probabilities = [discrete_power_law(n, alpha, beta) for alpha in elements]
    return probabilities


units = 100
x = np.arange(1, units+1)/units
y = discrete_power_law_distribution(units*2, 1.5)
plt.plot(x, y, label=r"$\beta$=1.5")
y = discrete_power_law_distribution(units*2, 2)
plt.plot(x, y, label=r"$\beta$=2")
y = discrete_power_law_distribution(units*2, 3)
plt.plot(x, y, label=r"$\beta$=3")
plt.xscale("log")
plt.legend()
plt.savefig("mutation_experiment/distribution_mutation_probability.png")
