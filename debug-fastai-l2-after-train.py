from fastai import *

from fastai.vision import *

bs = 16
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


path = untar_data(URLs.PETS); path


path.ls()


#%%
path_anno = path/'annotations'
path_img = path/'images'


#%%
fnames = get_image_files(path_img)
fnames[:5]

np.random.seed(2)
pat = r'/([^/]+)_\d+.jpg$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, 
    ds_tfms=get_transforms(), size=224, 
    num_workers=0, bs=bs).normalize(imagenet_stats)


learn = cnn_learner(data, models.resnet34, metrics=error_rate)

#%%
learn.load('stage-1');

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

#%%
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

#%%
interp.most_confused(min_val=2)
