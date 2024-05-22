"""
First time using Stan. 
Running a simple linear regression on Graduate Admissions Data.
See ./stan_regression.png for results.
"""

import numpy as np
import stan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme()

df = pd.read_csv('./admissions_data.csv')
x = df['GRE'].to_numpy()
y = df['admit_chance'].to_numpy()
N = len(x)

xy_df = pd.DataFrame({'gre': x, 'admit_chance': y})

data_dict = {'N': N, 'x': x, 'y': y}

lin_reg_model = """
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
"""

posterior = stan.build(lin_reg_model, data=data_dict)

fit = posterior.sample()

fit_df = fit.to_frame()

fit_df.to_csv('./lin_regression.csv')

fit_df = pd.read_csv('./lin_regression.csv')

alpha = np.mean(fit_df['alpha'])
beta = np.mean(fit_df['beta'])

xy_df['reg_line'] = alpha + beta*xy_df.gre

ax = sns.relplot(data = xy_df, x='gre', y='admit_chance')
ax.map_dataframe(sns.lineplot, 'gre', 'reg_line', color='orange')
ax.set(xlabel='GRE Score', ylabel='Admit Chance', title='Stan Regression Test')
plt.show()