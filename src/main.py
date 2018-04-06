from src.odl import OnlineDictionaryLearning
from src.DataGen import generate
from src.plot import plot_many

# Data parameters
data_type = "wave"
w_list = [0.02 * i for i in range(15)]
n_obs = 10000
interval = [0, 100]
split = 100
st_dev = 0.1
coefficients_range = [-2, 2]
sparsity = 0.8

# Learning parameters
it = 10000
lam = 0.001
dict_size = 10

# Generate data
data = generate(data_type, w_list, n_obs, interval, split, st_dev, coefficients_range, sparsity)

# Visualize data
x = [i for i in range(split)]
# plot_many(x, data, x_label="x", y_label="y")

# Learn the dictionaries
model = OnlineDictionaryLearning(data)
dic_predicted = model.learn(it, lam, dict_size)

plot_many(x, dic_predicted, x_label="x", y_label="predicted_dict",
          filename="predicted_dict-lam-" + str(lam) + str(st_dev), to_save=True)

x_2 = [i for i in range(len(model.cumulative_losses))]
plot_many(x_2, [model.cumulative_losses], x_label="Iterations", y_label="Cumulative losses",
            filename = "cumulative_losses-lam-" + str(lam), to_save=True)
