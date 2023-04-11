<p align="center" width="100%">
<img src="./assets/Stambecco logo.jpg" alt="Stambecco" style="width: 20%; min-width: 300px; display: block; margin: auto; border-radius: 140px;">
</p>

# Stambecco 🦌: Italian Instruction-following LLaMA Model

[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%20SA%204.0-orange.svg)](https://github.com/mchl-labs/stambecco/blob/main/DATA_LICENSE)

Stambecco is a Italian Instruction-following model based on the LLaMA model.
It comes in two versions: 7b and 13b parameters.
The entire project is built using **Google Colab**.

This repository contains code for reproducing the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) in Italian 🇮🇹 using [low-rank adaptation (LoRA)](https://arxiv.org/abs/2106.09685) provided by 🤗 Hugging Face's [PEFT](https://github.com/huggingface/peft) library. We provide an Instruct model of similar quality to text-davinci-003 that can run on a Raspberry Pi (for research), and the code is easily extended to the 13b.

To advance the state of the art of instruction-tuning for LLMs, we present the first attempt to use ***`GPT-4` generated instruction-following data*** for ***LLM finetuning in Italian***. Take a look at the results of the research: [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM#how-good-is-the-data)

In addition to the training code, which runs within hours on a single RTX 4090 or A100 40GB, we provide notebooks for downloading and inference on the foundation model and stambecco through a graphical interface using gradio.

## Acknowledgments

> If I have seen further it is by standing on the sholders of Giants.
> -- <cite>Isaac Newton</cite>

We started this section with this citation because everything we did was only possible due to the strong community and works that other people and groups did. For our work, we rely mainly on the works developed by: [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/), [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca Lora](https://github.com/tloen/alpaca-lora), [Cabrita](https://github.com/22-hours/cabrita), [Cleaned Alpaca Dataset](https://github.com/gururise/AlpacaDataCleaned), [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM), [Alpaca-LoRA-Serve](https://github.com/deep-diver/Alpaca-LoRA-Serve), [ChatGPT](https://openai.com/blog/chatgpt) and [Hugging Face](https://huggingface.co/). So, thank you all for the great work and share this with the world!

**Usage and License Notices**: Same as [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), Stambecco is intended and licensed for research use only. The dataset is CC BY NC SA 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

Please note that it is highly possible that the model output contains biased, conspiracist, offensive, or otherwise inappropriate and potentially harmful content. The model is intended for **research purposes only** and should be used with caution at your own risk. Production usage is not allowed.

## Data

We translated the [alpaca_data_cleaned.json](https://github.com/gururise/AlpacaDataCleaned/blob/main/alpaca_data_cleaned.json) and [alpaca_gpt4_data.json](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data.json) to Italian, adding checks (e.g. if the text is code -> don't translate it) to optimize the output of the translation. We recommend to use OpenAI `gpt-3.5-turbo` model to translate the dataset to reduce costs. Even this translation is not the best, the tradeoff between costs and results were. If you want to know more about how the dataset was built go to: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Cleaned Alpaca Dataset](https://github.com/gururise/AlpacaDataCleaned), [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM).


## Training

We finetuned the LLaMA model using PEFT from Hugging Face. To run the finetuning on Colab, on top of LLaMA-7B and LLaMA-13B, it is recommended an A100 40GB GPU. 
The datasets used for training are an Italian version of the GPT-4-LLM dataset, a dataset of `GPT-4` generated instruction-following data for the `plus` version of the models, and [mchl-labs/stambecco_data_it](https://huggingface.co/datasets/mchl-labs/stambecco_data_it) for the base version. 

See the model cards on Huggingface for more info about training hyperparameters.

## Play with 🦌 Stambecco models

**User Notice**: Facebook has not made the official LLaMA model weights open source, although various third-party download links are available online, such as `decapoda-research/llama-7b-hf` in the Hugging Face model library. It should be noted that the use of these links may not comply with Facebook's policies. Due to the reasons mentioned above, the project cannot release the complete weights of fine-tuned models. However, only the LoRA weights can be provided, which can be considered as a "patch" for the original LLaMA model.

The fine-tuned instruction-following Stambecco models are available on 🤗 Hugging Face:

- Fine-tuned Stambecco-7B model: [mchl-labs/stambecco-7b-plus](https://huggingface.co/mchl-labs/stambecco-7b-plus)
- Fine-tuned Stambecco-13B model: [mchl-labs/stambecco-13b-plus](https://huggingface.co/mchl-labs/stambecco-13b-plus)

You can infer these models by using the following Google Colab Notebook.

<a href="https://colab.research.google.com/github/mchl-labs/stambecco/blob/main/notebooks/Stambecco_demo.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


We released a translated dataset ([stambecco_data_it.json - Now also on Hugging Face's datasets](https://huggingface.co/datasets/mchl-labs/stambecco_data_it)), the models (available on the Hugging Face's hub) and the code to reproduce the results.

## 🖋️ Citations
```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}

@misc{selfinstruct,
  title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A. and Khashabi, Daniel and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2212.10560},
  year={2022}
}

@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}

@misc{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward and Shen, Yelong and Wallis, Phil and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Lu and Chen, Weizhu},
    year={2021},
    eprint={2106.09685},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}

@Misc{peft,
  title =        {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author =       {Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year =         {2022}
}

@article{peng2023gpt4llm,
    title={Instruction Tuning with GPT-4},
    author={Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao},
    journal={arXiv preprint arXiv:2304.03277},
    year={2023}
}
```

If stambecco inspires you and stambecco code, stambecco models or stambecco datasets are used in your research, please cite:

```
@misc{stambecco,
  author = {Michael},
  title = {Stambecco: Italian Instruction-following LLaMA Model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mchl-labs/stambecco}},
}
```
