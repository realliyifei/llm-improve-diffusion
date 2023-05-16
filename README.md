# Improving Text-to-Image Diffusion Generation via Large Language Model

## Environment

```
conda create -y -n diff python=3.9 cupy pkg-config compilers libjpeg-turbo opencv cudatoolkit=11.3 numba -c conda-forge
conda activate diff
pip install -r requirements.txt
```

Note: remember to fill in your OpenAI's API key in `main.py`

## Imagine-Then-Verbalize

Exectuion example:

```
python main.py desc.txt -p template_prompt
python visualizer.py desc.txt -p template_prompt 
```

Results:

* Template-based promting: [image folder](./images/visual_result/template_prompt) and [prompt folder](./prompts/template_prompt)
* CoT promting: [image folder](./images/visual_result/cot_prompt) and [prompt folder](./prompts/cot_prompt)

### Sketch-Then-Draw

Exectuion example:

```
python visualize_sketch.py
```

Results: [image folder](./images/visual_result/sketch) and [prompt folder](./prompts/sketch)