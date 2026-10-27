# Understanding Transformers

A self-study AI project exploring the **Transformer** architecture through hands-on experiments with Hugging Face models (text, audio and images).

- Notebook: [`transformel.ipynb`](transformel.ipynb) ([open in Colab](https://colab.research.google.com/github/perk-bevis/understanding-transformers/blob/main/transformel.ipynb))
- Live demo: [Hugging Face Space](https://huggingface.co/spaces/perkbevis/Transformel-models)

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
| `elmo.jpg`, `sea.jpg`, `white-flower.jpg`, `audio.mp3` | Sample inputs |

## Notes

- Models download on first use, so the first run of each tab is slow.
- The diffusion experiments in the notebook (Stable Diffusion, video) need a GPU and are not part of the app.

## References

- https://huggingface.co/
- https://arxiv.org/pdf/1910.01108 (DistilBERT)

# Link

https://huggingface.co/

https://arxiv.org/pdf/1910.01108

https://kaggle.com
