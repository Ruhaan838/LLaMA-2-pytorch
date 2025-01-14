<div align="center">

# LLaMA-2 - Let's Reproduce LLaMA from Scratch

<div style="display:flex; justify-content:center; gap: 20px;">

[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) [![arXiv](https://img.shields.io/badge/arXiv-LLaMA-B31B1B?logo=arXiv&logoColor=white)](https://arxiv.org/abs/2302.13971)


</div>

</div>

## Introduction

LLaMA (Large Language Model Meta AI) is an open-source language model it's has great benchmarks. This repository I will try to reproduce the LLaMA model from scrach with **training** and **inference**.

## Training
To train the model you need to specify the which model you need to train like **7B, 13B, 34B, 70B, T**
Here some note about the model

| Args/Params  | Dimension(d_model) | N Heads (num_heads) | N Layers(n_blocks) | Learning Rate(lr) |
|:------------:|:------------------:|:--------------------:|:-------------------:|:-----------------:|
|     7B       |        4096        |          32          |         32          |      3.0e-4       |
|    13B     |        5120        |          40          |         40          |      3.0e-4       |
|     34B      |        6656        |          52          |         60          |      1.5e-4       |
|     70B      |        8192        |          64          |         80          |      1.5e-4       |
|      T       |         768        |           8          |          4          |       3e-4        |


> If your Device is not able to fit the model before the loading the model <br>
> the code will caculate the size of the model and your device capacity  <br>
> that help you to not waste your time.

> One also IMP thing is that if none of this fit to your device than you can <br> run the 'T' mode of the model thats very basic model not have bigger size also


```bash
python train.py --model T --epoch 10
```

## args:
- --model [7B, 13B, 34B, 70B, T]
- --epoch Optional default = 1(int)



## Generating Texts

I am still working on this like if I can load the pretrain weigths of the model that can generate the text without any kind of writing the same code in the huggingface.


## Citation

```bibtex
@article{touvron2023llama,
  title   = "{LLaMA}: Open and Efficient Foundation Language Models",
  author  = "Hugo Touvron",
  journal = "arXiv preprint arXiv:2302.13971",
  year    = "2023",
  month   = "Feb",
  url     = "https://arxiv.org/pdf/2302.13971"
}
```


## File Structure

```markdown
├LLaMA
├── extra
│   ├── KVcachemultihead.py
│   ├── multiheadAttention.py
│   ├── multiqueryAttentionKVcache.py
│   ├── README.md
│   └── RotaryPE.py
├── model
│   ├── __pycache__
│   ├── __init__.py
│   ├── config.py
│   ├── embedding.py
│   └── model.py
├── dataset.py
├── generate.py
├── README.md
├── requirements.txt
├── train.py
└── .gitignore
```
