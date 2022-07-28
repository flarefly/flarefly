'''
Macro to test data handling
'''
from flarefly.data_handler import DataHandler
import numpy as np
import zfit
import matplotlib.pyplot as plt
import pandas as pd

# Test data handler on numpy array
mu_true = 0
sigma_true = 1

data_np = np.random.normal(mu_true, sigma_true, size=10000)
data_to_fit = DataHandler(data_np, var_name='x', limits=[-5, 5])

mu = zfit.Parameter('mu', 1, -3, 3, step_size=0.2)
sigma_num = zfit.Parameter('sigma42', 1, 0.1, 10, floating=False)
gauss = zfit.pdf.Gauss(obs=data_to_fit.get_obs(), mu=mu, sigma=sigma_num)

# Create the negative log likelihood
obs = data_to_fit.get_obs()
data = data_to_fit.get_data()
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)  # loss

# Load and instantiate a minimizer
minimizer = zfit.minimize.Minuit()
minimum = minimizer.minimize(loss=nll)
params = minimum.params
n_bins = 50
range_ = (-5,5)
_ = plt.hist(data_to_fit.get_data(input_data=True), bins=n_bins, range=range_)
x = np.linspace(*range_, num=1000)
pdf = zfit.run(gauss.pdf(x))
test_plot = plt.plot(x, data_np.shape[0] / n_bins * (range_[1] - range_[0]) * pdf)
plt.savefig('test_plot.png')

# Test data handler on pandas dataframe
df = data_to_fit.get_data_to_pandas()
print(df.head())

data_to_fit = DataHandler(df, var_name='x', limits=[-5, 5])

obs = data_to_fit.get_obs()
data = data_to_fit.get_data()

_ = plt.hist(data_to_fit.get_data(input_data=True), bins=n_bins, range=range_)
x = np.linspace(*range_, num=1000)
pdf = zfit.run(gauss.pdf(x))
test_plot = plt.plot(x, data_np.shape[0] / n_bins * (range_[1] - range_[0]) * pdf)
plt.savefig('test_plot_pd.png')
