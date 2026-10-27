<div align="center">

<img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="90">

<h1>Understanding Transformers</h1>

<p><b>Learn by doing, understand by building:</b><br>
explore the Transformer architecture through hands-on experiments with text, audio and images.</p>

<p>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT"></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12-3776AB" alt="Python"></a>
<a href="https://huggingface.co/spaces/perkbevis/Transformel-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-FFD21E" alt="Hugging Face Space"></a>
<a href="https://colab.research.google.com/github/perk-bevis/understanding-transformers/blob/main/transformel.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

<p>
<a href="https://huggingface.co/spaces/perkbevis/Transformel-models">Live Demo</a> ·
<a href="transformel.ipynb">Notebook</a> ·
<a href="#whats-inside">What's inside</a> ·
<a href="#run-locally">Run locally</a> ·
<a href="#references">References</a>
</p>

</div>

---

## What's inside

- **GPT-2:** greedy, top-k, top-p, temperature and random sampling, plus next-token probabilities
- **Tokenizer:** how GPT-2 splits words into tokens
- **Sentiment analysis:** DistilBERT (general) and FinBERT (financial)
- **Named entity recognition**, **question answering** and **machine translation** (NLLB-200)
- **Audio:** waveform and mel spectrogram, classification (AST), speech-to-text, text-to-speech
- **Image processing:** RGB channels, grayscale and resizing

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

`ffmpeg` is needed for reading mp3 files.

## Files

| File | Description |
|---|---|
| `transformel.ipynb` | Experiments and notes |
| `app.py` | Gradio demo app |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages (for Spaces) |
| `elmo.jpg`, `sea.jpg`, `white-flower.jpg` | Sample images |
| `Hugging-Face-Course_Resources_audio.mp3` | Sample audio |

## Notes

- Models download on first use, so the first run of each tab is slow.
- The diffusion experiments in the notebook (Stable Diffusion, video) need a GPU and are not part of the app.

## References

- [Hugging Face](https://huggingface.co/)
- [DistilBERT paper](https://arxiv.org/pdf/1910.01108)
- [Kaggle](https://kaggle.com)
