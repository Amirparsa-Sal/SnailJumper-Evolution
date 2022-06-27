import pickle
import matplotlib.pyplot as plt

data = pickle.load(open('data/evolution.pkl', 'rb'))
x = [i for i in range(len(data['max_fitness']))]
plt.plot(x, data['max_fitness'], label='Max Fitness')
plt.plot(x, data['avg_fitness'], label='Avg Fitness')
plt.plot(x, data['min_fitness'], label='Min Fitness')
plt.legend()
plt.show()