
#%%
from fastai.vision import *

#%% [markdown]
# ## Getting the data
# The planet dataset isn't available on the fastai dataset page due to copyright restrictions. You can download it from Kaggle however. Let's see how to do this by using the Kaggle API as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.
# 
# First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal (depending on your platform you may need to modify this slightly to either add source activate fastai or similar, or prefix pip with a path. Have a look at how conda install is called for your platform in the appropriate Returning to work section of https://course.fast.ai/. (Depending on your environment, you may also need to append "--user" to the command.)

#%%
# ! pip install kaggle --upgrade

#%% [markdown]
# Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.
# 
# Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal). For Windows, uncomment the last two commands.

#%%
path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)
path


#%% [markdown]
# issue: SyntaxError: invalid syntax
# 
# solution: https://github.com/Kaggle/kaggle-api/issues/12
# 
# 
# issue: 403 Forbidden:
# 
# solution: Join Competition and Accept rules
# 
# https://github.com/Kaggle/kaggle-api/issues/87 

#%%



