import os
import argparse
from tqdm import tqdm
import utils
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import textwrap

def save_subplot(desc1, desc2, folder1, folder2, filename, save_folder):
    '''Create a plot where the left is 2x2 subplot of images in folder1, with title desc1, 
                       and the right is 2x2 subplot of images in folder2, with title desc2.
       Save as save_folder/filename.png.
       PS1. This will rewrite the current visualization if there is one.
       PS2. This will only plot the first 4 images in folder1 and folder2.
    '''
    # Get image file paths
    image_files1 = [file for file in os.listdir(folder1) if file.endswith(('.png', '.jpg', '.jpeg'))]
    image_files2 = [file for file in os.listdir(folder2) if file.endswith(('.png', '.jpg', '.jpeg'))]

    fig = plt.figure(figsize=(13, 6))
    gs1 = gridspec.GridSpec(2, 2, left=0.05, right=0.48, wspace=0.02, hspace=0.02) # set the spacing between axes.
    for i, (file, ax) in enumerate(zip(image_files1[:4], gs1)):
        image_path = os.path.join(folder1, file)
        image = Image.open(image_path)
        ax = plt.subplot(gs1[i])
        ax.imshow(image)
        ax.axis('off')
    
    gs2 = gridspec.GridSpec(2, 2, left=0.52, right=0.95, wspace=0.02, hspace=0.02) # set the spacing between axes.
    for i, (file, ax) in enumerate(zip(image_files2[:4], gs2)):
        image_path = os.path.join(folder2, file)
        image = Image.open(image_path)
        ax = plt.subplot(gs2[i])
        ax.imshow(image)
        ax.axis('off')

    # Set the subtitles
    wrapped_desc1 = textwrap.fill(desc1, width=50)
    wrapped_desc2 = textwrap.fill(desc2, width=50) # Can show up to five lines
    ax_left = fig.add_subplot(gs1[:])
    ax_left.axis('off')
    ax_left.set_title(wrapped_desc1, fontsize=16)
    ax_right = fig.add_subplot(gs2[:])
    ax_right.axis('off')
    ax_right.set_title(wrapped_desc2, fontsize=16)

    # Save the plot
    plt.savefig(os.path.join(save_folder, f"{filename}.png"))
    plt.close()


def visualize(args):
    # Ensure save_folder exists
    print(f"Saving visualization to {args.save_folder}...")
    os.makedirs(args.save_folder, exist_ok=True)
    for folder in os.listdir(args.image_folder):
        folder1 = os.path.join(args.image_folder, folder, "raw")
        folder2 = os.path.join(args.image_folder, folder, "sketched")
        title = folder.replace("_", " ").title()
        desc1 = f"(Raw) {title}"
        desc2 = f"(Sketched) {title}"
        save_subplot(desc1, desc2, folder1, folder2, folder, args.save_folder)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subplot images of descriptions and prompts")
    parser.add_argument("-i", "--image_folder", default="images", help="Path to the output image folder for storing images")
    parser.add_argument("-n", "--num_images", type=int, default=4, help="Number of images to plot per description")
    parser.add_argument("-s", "--save_folder", default="images/visual_result", help="Path to the visualizaton folder")
    args = parser.parse_args()
    args.image_folder = os.path.join(args.image_folder, "sketch")
    args.save_folder = os.path.join(args.save_folder, "sketch")
    visualize(args)
