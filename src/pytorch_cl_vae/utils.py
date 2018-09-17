import seaborn
seaborn.set()
import matplotlib.pyplot as plt

def plot_latent(z, labels):
    seaborn.scatterplot(x=z[:, 0], y=z[:, 1], hue=labels,  legend="full", palette='Set2')

