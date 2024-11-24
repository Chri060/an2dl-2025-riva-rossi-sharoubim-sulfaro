import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
import os

LABELS = {0: 'Basophil', 1: 'Eosinophil', 2: 'Erythroblast', 3: 'Immature granulocytes', 4: 'Lymphocyte', 5: 'Monocyte', 6: 'Neutrophil', 7: 'Platelet'}

SEED = 42

# Seaborn settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)


## DATASET UTILS

# Load a Dataset from the dataset folder, returns images X and labels y
def load_dataset(name='cleaned'):
    data = np.load(f'dataset/{name}.npz')
    return data['images'], data['labels']

# Load cleaned dataset
def load_cleaned_dataset():
    return load_dataset(name='cleaned_set')

# Load class-equalized dataset using mixup augmentation
# Type : oversampled | mixup
def load_balanced_dataset(type):
    return load_dataset(name=f'balanced_{type}_set')

# Splits the Dataset into 3 sets given the percentages of test and validation compared to the entire dataset size
def split_dataset(X,y, test_size=0.2, val_size=0.2):
    ts = round(X.shape[0] * test_size)
    vs = round(X.shape[0] * val_size)
    if test_size != 0.0 : X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=SEED)
    else : X_train, y_train, X_test, y_test = X, y, [], []
    if val_size != 0.0 : X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=vs, random_state=SEED)
    else : X_val, y_val = [], []
    return X_train, X_val, X_test, y_train, y_val, y_test


## PLOTTING UTILS -----------

# Stores the last plotted image in the images folder
# If the image already exists it doesn't overwrite it unless force=True
def save_fig(name, force=False):
    f = 'png'
    if not force and os.path.exists(f'images/{name}.{f}') :
        print('Image already saved.')
        return
    plt.savefig(f'./images/{name}.{f}',dpi=300,format=f,bbox_inches='tight')
    print ('Image saved') if force==False else print('Image force-saved')

# Given a set of images it creates a table with xlabel as its LABEL
def plot_images(images, targets=[], print_idx=False, figsize=(10,10), rows=2, columns=1, fontsize=10, rand=True):
    if (rows*columns > images.shape[0]) :
        raise Exception(f"too few images for {rows} rows and {columns} columns")
    fig = plt.figure(figsize=figsize)
    if rand: indexes = np.random.randint(low=0, high=images.shape[0], size=rows*columns)
    else: indexes = range(images.shape[0])
    for i, j in enumerate(indexes):
        ax = fig.add_subplot(rows,columns, i+1)
        ax.imshow(images[j])
        if len(targets)!=0:
            if print_idx:
                ax.set_xlabel(f"{j + 1}\n{LABELS[targets[j].item()]}", fontsize=fontsize)
            else : 
                ax.set_xlabel(f"{LABELS[targets[j].item()]}", fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

# Plot class distribution comparison between train and test
def plot_class_distribution(y,y_val=[] ,y_test=[]):
    # Set seaborn style for the plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(18, 6))

    # Calculate class distributions for training and test sets
    train_dist = np.bincount(y)
    if any(y_val): val_dist = np.bincount(y_test)
    if any(y_test): test_dist = np.bincount(y_test)

    # Create x positions and set bar width
    x = np.arange(len(LABELS))
    width = 0.27

    # Plot bars for training and test distributions
    plt.bar(x - width , train_dist, width, label='Training', color='#2ecc71', alpha=0.7)
    if any(y_val): plt.bar(x, val_dist, width, label='Validation', color='#3498ef', alpha=0.7)
    if any(y_test): plt.bar(x + width, test_dist, width, label='Test', color='#3498db', alpha=0.7)

    # Customise plot title and labels
    plt.title('Class Distribution', pad=20, fontsize=14)
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')

    # Set class names as x-axis labels with rotation
    plt.xticks(x, LABELS.values(), rotation=45, ha='right')

    # Add legend for training and test distributions
    plt.legend(loc='lower right')

    # Adjust layout for optimal spacing
    plt.tight_layout()
    plt.show()

# Plots training history
def plot_history(history):
    plt.figure(figsize=(15,5))
    plt.plot(history['loss'], label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_loss'], label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Categorical Crossentropy')
    plt.grid(alpha=.3)
    
    plt.figure(figsize=(15,5))
    plt.plot(history['accuracy'], label='Training', alpha=.3, color='#ff7f0e', linestyle='--')
    plt.plot(history['val_accuracy'], label='Validation', alpha=.8, color='#ff7f0e')
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    plt.grid(alpha=.3)
    
    plt.show()