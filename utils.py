import os
import re
from tqdm import tqdm

def sanitize_folder_name(folder_name):
    return re.sub(r'[^\w\-_]', '_', folder_name).strip().lower()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def read_descriptions(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

def get_next_image_number(output_folder):
    existing_images = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    if not existing_images:
        return 1
    max_number = max([int(re.search(r'img_(\d+).png', img).group(1)) for img in existing_images])
    return max_number + 1

def generate_and_save_images(description, output_folder, num_images, rewrite, pipe, image_model, verbose=True):
    create_folder_if_not_exists(output_folder)
    prompt = [description] * num_images
    images = None
    if image_model=='sd':
        images = pipe(prompt).images
    elif image_model=='dalle2':
        pass
    start_number = 1 if rewrite else get_next_image_number(output_folder)
    for i, image in enumerate(images, start=start_number):
        image.save(os.path.join(output_folder, f'img_{i}.png'))
    if verbose:
        print(f"Generated {num_images} images to '{output_folder}'.")

def read_prompt_template_from_file(filename):
    with open(filename, 'r') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    return prompts

def get_new_desc_by_txt_file(input_file):
    with open(input_file) as f:
        lines = f.readlines()
    new_desc = next((line.strip() for line in reversed(lines) if line.strip()), None)
    if "One-sentence description" in new_desc:
        new_desc = new_desc.split(":")[1].strip()
    return new_desc

