import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Visualization utilities

# converts gray mask (with 12 classes) to rgb
def colorize12_mask(mask,colormap):
  rgb_mask_shape = mask.shape + (3,)
  rgb_mask = np.zeros(rgb_mask_shape, dtype=np.uint8)

  for idx, (label, colors) in enumerate(colormap.items()):
    rgb_mask[mask == idx] = colors[0]
  return rgb_mask

# converts gray mask (with 32 classes) to rgb
def colorize_mask(mask,colormap):
  rgb_mask_shape = mask.shape + (3,)
  rgb_mask = np.zeros(rgb_mask_shape, dtype=np.uint8)

  for label, color in colormap.items():
    rgb_mask[mask == label] = color
  return rgb_mask

# Shows a mask superposed on its original image
def show_samples_superposed(dataset, nb_samples, random=False):
  fig, axes = plt.subplots(1,nb_samples, figsize=(8*nb_samples,4))

  if random:
    samples = np.random.choice(len(dataset),nb_samples, replace= False)
  else:
    samples = range(nb_samples)
  
  for idx, sample in enumerate(samples):
    input = dataset[sample]
    image, mask = input['pixel_values'], input['labels'] # (c,h,w), (h,w,c)
    image = np.transpose(image,(1,2,0))

    if dataset.num_classes == 32 :
      rgb_mask = colorize_mask(mask, dataset.class_colors)
    else:
      rgb_mask = colorize12_mask(mask, dataset.class_colors)
    ims1 = axes[idx].imshow(image)
    ims2 = axes[idx].imshow(rgb_mask,alpha=0.7)

# Shows images and their labels separately
def show_samples(dataset, nb_samples, random=False):
  fig, axes = plt.subplots(2,nb_samples, figsize=(8*nb_samples,8))

  if random:
    samples = np.random.choice(len(dataset),nb_samples, replace= False)
  else:
    samples = range(nb_samples)
  
  for idx, sample in enumerate(samples):
    input = dataset[sample]
    image, mask = input['pixel_values'], input['labels'] # (c,h,w), (h,w,c)
    image = np.transpose(image,(1,2,0))

    if dataset.num_classes == 32 :
      rgb_mask = colorize_mask(mask, dataset.class_colors)
    else:
      rgb_mask = colorize12_mask(mask, dataset.class_colors)
    ims1 = axes[0,idx].imshow(image)
    axes[0,idx].set_title("Image")
    ims2 = axes[1,idx].imshow(rgb_mask,alpha=0.7)
    axes[0,idx].set_title("Ground Truth")

# Gather statistics about the data

# counts number of samples in each class
def compute_class_distribution(dataset):
  summary = [0]*dataset.num_classes

  for inputs in dataset:
      mask = inputs['labels']
      labels, counts = np.unique(mask, return_counts=True)
      for idx,label in enumerate(labels):
        summary[label] += counts[idx]
  return summary

# Plots classes distribution
def plot_class_distribution(id2label,summary): 
  
  plt.figure(figsize=(18,10))
  sns.barplot(y=summary, x= list(id2label))
  plt.axes().set_xticklabels(list(id2label), rotation=90)
  plt.show()

# computes the weight of each class
def compute_class_weights(total, class_counts):
  weights = []
  for class_count in class_counts:
    weights.append(total/class_count)
  return weights