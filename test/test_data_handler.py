'''
Macro to test data handling
'''
from re import T
from flarefly.data_handler import DataHandler
import numpy as np
import zfit
import matplotlib.pyplot as plt
import pandas as pd
from flarefly.utils import Colour_print

def plotTest(data_to_fit, n_bins, range_, label, input_data=True):
    _ = plt.hist(data_to_fit.get_data(input_data), bins=n_bins, range=range_)
    x = np.linspace(*range_, num=1000)
    pdf = zfit.run(gauss.pdf(x))
    test_plot = plt.plot(x, data_to_fit.get_data(input_data).shape[0] / n_bins * (range_[1] - range_[0]) * pdf)
    plt.savefig(f'test_plot_{label}.png')
    plt.close()
    return test_plot

# Test data handler on numpy array
mu_true = -1
sigma_true = 1

data_np = np.random.normal(mu_true, sigma_true, size=10000)
data_to_fit = DataHandler(data_np, var_name='x', limits=[-5, 5])
obs = data_to_fit.get_obs()
data = data_to_fit.get_data()

mu = zfit.Parameter('mu', 1, -3, 3, step_size=0.2)
sigma_num = zfit.Parameter('sigma42', 1, 0.1, 10, floating=False)
gauss = zfit.pdf.Gauss(obs=obs, mu=mu, sigma=sigma_num)

# Create the negative log likelihood
nll = zfit.loss.UnbinnedNLL(model=gauss, data=data)  # loss
minimizer = zfit.minimize.Minuit()
minimum = minimizer.minimize(loss=nll)
params = minimum.params
n_bins = 100
range_ = (-5,5)
test_plot = plotTest(data_to_fit, n_bins, range_, 'numpyarray')
Colour_print('Test data handler for numpy array passed', 'OK')

# Test data handler on pandas dataframe
df = data_to_fit.dump_to_pandas()
data_to_fit = DataHandler(df, var_name='x', limits=[-5, 5])
obs = data_to_fit.get_obs()
data = data_to_fit.get_data()
plotTest(data_to_fit, n_bins, range_, 'pandas')
Colour_print('Test data handler for pandas dataframe passed', 'OK')

# Test data handler on ROOT histogram
data_to_fit = DataHandler('/home/stefano/Desktop/cernbox/flarefly/hTest.root',
                          var_name='x', limits=[-5, 5], histoname='h1')
obs = data_to_fit.get_obs()
data = data_to_fit.get_data(True)
print(data)
plt.bar((data[1][1:] + data[1][:-1]) * .5, data[0], width=(data[1][1] - data[1][0]))
plt.savefig('test_plot_root.png')
Colour_print('Test data handler for ROOT histogram passed', 'OK')
