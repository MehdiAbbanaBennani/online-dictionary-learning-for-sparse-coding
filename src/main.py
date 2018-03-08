from src.odl import OnlineDictionaryLearning
from src.DataGen import generate
from src.plot import plot_many

# Data parameters
data_type = "wave"
w_list = [0.01 * i for i in range(10)]
w_counts = [20 for i in range(10)]
interval = [0, 100]
split = 100
st_dev = 0.1

# Learning parameters
it = 10
lam = 0.1
dict_size = 5

# Generate data
data = generate(data_type, w_list, w_counts, interval, split, st_dev)

# Visualize data
x = [i for i in range(split)]
# plot_many(x, data, x_label="x", y_label="y")

# Learn the dictionaries
model = OnlineDictionaryLearning(data)
dic_predicted = model.learn(it, lam, dict_size)

plot_many(x, dic_predicted, x_label="x", y_label="predicted_dict")