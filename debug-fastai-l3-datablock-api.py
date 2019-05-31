
# ## The data block API
# The data block API lets you customize the creation of a DataBunch by isolating the underlying parts of that process in separate blocks, mainly:
# 
# Where are the inputs and how to create them?
# How to split the data into a training and validation sets?
# How to label the inputs?
# What transforms to apply?
# How to add a test set?
# How to wrap in dataloaders and create the DataBunch?
# Each of these may be addressed with a specific block designed for your unique setup. Your inputs might be in a folder, a csv file, or a dataframe. You may want to split them randomly, by certain indices or depending on the folder they are in. You can have your labels in your csv file or your dataframe, but it may come from folders or a specific function of the input. You may choose to add data augmentation or not. A test set is optional too. Finally you have to set the arguments to put the data together in a DataBunch (batch size, collate function...)
# 
# The data block API is called as such because you can mix and match each one of those blocks with the others, allowing for a total flexibility to create your customized DataBunch for training, validation and testing. The factory methods of the various DataBunch are great for beginners but you can't always make your data fit in the tracks they require.
# 
# Mix and match
# 
# As usual, we'll begin with end-to-end examples, then switch to the details of each of those parts.


# ### Examples of use
# Let's begin with our traditional MNIST example.

from fastai.vision import *


path = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)
path.ls()


(path/'train').ls()


data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=64)

data = (ImageList.from_folder(path) #Where to find the data? -> in path and its subfolders
        .split_by_folder()              #How to split in train/valid? -> use the folders
        .label_from_folder()            #How to label? -> depending on the folder of the filenames
        .add_test_folder()              #Optionally add a test set (here default name is test)
        .transform(tfms, size=64)       #Data augmentation? -> use tfms with a size of 64
        .databunch())                   #Finally? -> use the defaults for conversion to ImageDataBunch


data.train_ds[0]

# ## Another Example 
# Let's look at another example from vision.data with the planet dataset. This time, it's a multiclassification problem with the labels in a csv file and no given split between valid and train data, so we use a random split. The factory method is:

planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


pd.read_csv(planet/"labels.csv").head()


data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', label_delim = ' ', ds_tfms=planet_tfms)


planet.ls()


pd.read_csv(planet/"labels.csv").head()


data = (ImageList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        #Where to find the data? -> in planet 'train' folder
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_df(label_delim=' ')
        #How to label? -> use the second column of the csv file and split the tags by ' '
        .transform(planet_tfms, size=128)
        #Data augmentation? -> use tfms with a size of 128
        .databunch())                          
        #Finally -> use the defaults for conversion to databunch


# ## Create databunch from raw data
# The data block API also allows you to get your data together in problems for which there is no direct ImageDataBunch factory method. For a segmentation task, for instance, we can use it to quickly get a DataBunch. Let's take the example of the camvid dataset. The images are in an 'images' folder and their corresponding mask is in a 'labels' folder.


camvid = untar_data(URLs.CAMVID_TINY)
path_lbl = camvid/'labels'
path_img = camvid/'images'


# We have a file that gives us the names of the classes (what each code inside the masks corresponds to: a pedestrian, a tree, a road...)

codes = np.loadtxt(camvid/'codes.txt', dtype=str); codes

# And we define the following function that infers the mask filename from the image filename.

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

vdata = (SegmentationItemList.from_folder(path_img)
        #Where to find the data? -> in path_img and its subfolders
        .split_by_rand_pct()
        #How to split in train/valid? -> randomly with the default 20% in valid
        .label_from_func(get_y_fn, classes=codes)
        #How to label? -> use the label function on the file name of the data
        .transform(get_transforms(), tfm_y=True, size=128)
        #Data augmentation? -> use tfms with a size of 128, also transform the label images
        .databunch())
        #Finally -> use the defaults for conversion to databunch






