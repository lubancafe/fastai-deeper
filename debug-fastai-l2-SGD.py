

#get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.basics import *

# In this part of the lecture we explain Stochastic Gradient Descent (SGD) which is an optimization method commonly used in neural networks. We will illustrate the concepts with concrete examples.

# ## Linear Regression problem
# The goal of linear regression is to fit a line to a set of points.

n=100


x = torch.ones(n,2) 
x[:,0].uniform_(-1.,1)
x[:5]


#%%

a = tensor(3.,2); a


#%%
y = x@a + torch.rand(n)

#%% [markdown]
# In python, @ is matrix production

#%%
plt.scatter(x[:,0], y);

#%% [markdown]
# You want to find parameters (weights) a such that you minimize the error between the points and the line x@a. Note that here a is unknown. For a regression problem the most common error function or loss function is the mean squared error.

#%%



