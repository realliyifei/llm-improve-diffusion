import os
import torch
import argparse
from datetime import datetime
from tqdm import tqdm
import utils

os.environ["HF_HOME"] = "/nlp/data/diff"
from diffusers import StableDiffusionPipeline

import openai
# from dalle2 import Dalle2
openai.api_key = "your-api-key"
# dalle = Dalle2("your-api-key")

### Prompting 
def template_prompting(prompts, sanitized_description, folder, overwrite, verbose=True):
    full_output = ""

    for prompt in prompts:
        full_output += f"> {prompt}\n\n"
        response = openai.Completion.create(engine="text-davinci-003", prompt=full_output, 
                                            max_tokens=100, n=1, stop=None, temperature=0.7)

        response_text = response.choices[0].text.strip()
        full_output += f"{response_text}\n\n"

    # Save full response to txt file in the specified folder
    txt_file = f"{folder}/{sanitized_description}.txt" 
    if os.path.exists(txt_file) and not overwrite:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_file = f"{folder}/{sanitized_description}_{timestamp}.txt"
    with open(txt_file, "w") as f:
        f.write(full_output)
    # Save full response to txt file in the specified folder
    txt_file = f"{folder}/{sanitized_description}.txt" 
    if os.path.exists(txt_file) and not overwrite:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_file = f"{folder}/{sanitized_description}_{timestamp}.txt"
    with open(txt_file, "w") as f:
        f.write(full_output)
        
    if verbose:
        print(f"Saved as '{txt_file}'.")

    print("New description:", response_text)
    return response_text


def cot_prompt(prompt, sanitized_description, folder, overwrite=False, verbose=True):
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, 
                                        max_tokens=1000, n=1, stop=None, temperature=0.7)
    response_text = response.choices[0].text.strip()
    
    
    # Save full response to txt file in the specified folder
    txt_file = f"{folder}/{sanitized_description}.txt" 
    if os.path.exists(txt_file) and not overwrite:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        txt_file = f"{folder}/{sanitized_description}_{timestamp}.txt"
    with open(txt_file, "w") as f:
        f.write(response_text)
    
    if verbose:
        print(f"Saved as '{txt_file}'.")

    # Extract the one-sentence description
    for line in response_text.split("\n"):
        if line.startswith("One-sentence description:"):
            new_desc = line.split("One-sentence description:")[1].strip()
            # print("New description:", new_desc)
            return new_desc

### Main
def main(args):
    input_file = args.input_file
    image_model = args.image_model
    image_folder = args.image_folder
    prompt_folder = args.prompt_folder
    num_images = args.num_images
    prompt_type = args.prompt_type
    overwrite = args.overwrite

    utils.create_folder_if_not_exists(image_folder)

    # Require more than 10GB of GPU RAM for the default float 32 precision
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
    generator = torch.Generator("cuda").manual_seed(1024)  # deterministic result

    descriptions = utils.read_descriptions(input_file)
    if prompt_type == "raw": # raw description
        for description in tqdm(descriptions, desc="Generating images"):
            sanitized_description = utils.sanitize_folder_name(description)
            subfolder_image = os.path.join(image_folder, prompt_type, sanitized_description)
            utils.generate_and_save_images(description, subfolder_image, num_images, overwrite, pipe)
    else: # description with template-based or cot prompt 
        prompt_template_file = f"{prompt_folder}/{prompt_type}/template.txt"
        prompt_templates = utils.read_prompt_template_from_file(prompt_template_file)
        subfolder_prompt = os.path.join(prompt_folder, prompt_type, "results")
        utils.create_folder_if_not_exists(subfolder_prompt)
        for description in tqdm(descriptions, desc="Prompting descriptions and generating images"):
            formatted_prompts = [pt.format(description) for pt in prompt_templates]
            sanitized_description = utils.sanitize_folder_name(description)
            new_description = ""
            if 'cot' in prompt_type:
                formatted_prompts = "\n".join(formatted_prompts)
                new_description = cot_prompt(formatted_prompts, sanitized_description, subfolder_prompt, overwrite)
            else:
                new_description = template_prompting(formatted_prompts, sanitized_description, subfolder_prompt, overwrite)
            subfolder_image = os.path.join(image_folder, prompt_type, sanitized_description)
            utils.generate_and_save_images(new_description, subfolder_image, num_images, overwrite, pipe, image_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from descriptions in a text file")
    parser.add_argument("input_file", help="Path to the input text file containing descriptions")
    parser.add_argument("-m", "--image_model",  default='sd', choices=['sd', 'dalle2'], help="Image generation model to use") # Note that we only developed the code for sd
    parser.add_argument("-i", "--image_folder", default="images", help="Path to the output image folder for storing images")
    parser.add_argument("-n", "--num_images", type=int, default=4, help="Number of images to generate per description")
    parser.add_argument("-f", "--prompt_folder", default="prompts", help="Path to the GPT prompt folder for prompt candidates")
    parser.add_argument("-p", "--prompt_type", default="raw", help="The type of prompt to use")
    parser.add_argument("-o", "--overwrite", action='store_true', 
                        help="Overwrite images (and prompt results) if set, otherwise continue numbering after existing images (and prompt results with timestamp)")
    args = parser.parse_args()
    main(args)
