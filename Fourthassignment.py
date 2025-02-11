import numpy as np
import matplotlib.pyplot as plt

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    sums = dice1 + dice2
    h, h2 = np.histogram(sums, range(2, 14))
    plt.bar(h2[:-1], h / n, width=0.8, alpha=0.5, color='Green' )
    plt.title(f"Histogram of Dice Sums for n={n}")
    plt.xlabel("Sum of Dice")
    plt.ylabel("Frequency")
    plt.show()

""" Answer no 4: 
When running the simulation for increasing values of n, you observe that for smaller n (e.g., 500 or 1000),
the histogram of dice sums shows noticeable fluctuations and does not match the theoretical probabilities well. 
As n increases (e.g., 5000 or 10000), the histogram stabilizes, and the frequencies of the sums align more 
closely with the expected probabilities. For very large n (e.g., 100000), the histogram almost perfectly matches
the theoretical distribution. This demonstrates the law of large numbers, where increasing the sample size reduces
randomness and brings observed results closer to the expected values."""

""" Answer no 5:
This phenomenon is related to regression to the mean, which describes how extreme or unusual outcomes
in small samples tend to move closer to the average as the sample size increases. In the dice simulation, 
for small n, the frequencies of sums like 2 or 12 may deviate significantly from their theoretical 
probabilities due to randomness. As n grows, these frequencies regress toward the mean, stabilizing and 
aligning with the expected probabilities. This illustrates how larger sample sizes reduce the impact of randomness, 
ensuring results converge to the expected distribution.
"""