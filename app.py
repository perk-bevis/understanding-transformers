"""
Understanding Transformers - demo app (Gradio)
Turns the experiments in `transformel.ipynb` into an interactive app:
GPT-2 (sampling), Tokenizer, Sentiment, NER, QA, Translation, Audio, Image processing.

Models are loaded lazily (on first use) and cached, so the Space starts quickly.
"""
import glob
import os
from functools import lru_cache

import gradio as gr
import librosa
import librosa.display
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from PIL import Image
from transformers import (
    ASTForAudioClassification,
    AutoFeatureExtractor,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def asset(name: str) -> str:
    return os.path.join(BASE_DIR, name)


def find_audio():
    """Find the sample mp3 (audio.mp3 on the Space, or *audio.mp3 on GitHub)."""
    for pattern in ("audio.mp3", "*audio*.mp3"):
        hits = glob.glob(os.path.join(BASE_DIR, pattern))
        if hits:
            return hits[0]
    return None


AUDIO_SAMPLE = find_audio()
IMAGE_SAMPLES = [
    [asset(n)] for n in ("elmo.jpg", "sea.jpg", "white-flower.jpg") if os.path.exists(asset(n))
]


# =====================================================================
# 1. GPT-2: sampling strategies (greedy / top-k / top-p / temperature / random)
# =====================================================================
@lru_cache(maxsize=1)
def load_gpt2():
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
    return tok, model


def greedy_decode(logits):
    return torch.argmax(logits, dim=-1).reshape(1)


def top_k_sampling(logits, k=50):
    values, indices = torch.topk(logits, int(k))
    probs = F.softmax(values, dim=-1)
    return indices[torch.multinomial(probs, 1)]


def top_p_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Shift the cumulative sum by one step so the first token that crosses the
    # threshold is always kept (otherwise, if the top token already has
    # probability > p, every logit gets masked and the result is NaN).
    mask = (cumulative - sorted_probs) > p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    filtered = F.softmax(sorted_logits, dim=-1)
    return sorted_indices[torch.multinomial(filtered, 1)]


def temperature_sampling(logits, temperature=1.0):
    probs = F.softmax(logits / max(float(temperature), 1e-4), dim=-1)
    return torch.multinomial(probs, 1)


def random_sampling(logits):
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


def pick_token(logits, strategy, k, p, temperature):
    if strategy == "Greedy":
        return greedy_decode(logits)
    if strategy == "Top-k":
        return top_k_sampling(logits, k)
    if strategy == "Top-p (Nucleus)":
        return top_p_sampling(logits, p)
    if strategy == "Temperature":
        return temperature_sampling(logits, temperature)
    return random_sampling(logits)


def generate(prompt, strategy, n_tokens, k, p, temperature, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    tok, model = load_gpt2()
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        for _ in range(int(n_tokens)):
            logits = model(ids).logits[0, -1]
            nxt = pick_token(logits, strategy, k, p, temperature)
            ids = torch.cat([ids, nxt.reshape(1, 1)], dim=1)
            if int(nxt) == tok.eos_token_id:
                break
    return tok.decode(ids[0])


def next_token_probs(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a sentence.")
    tok, model = load_gpt2()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=10)
    return [[repr(tok.decode(int(i))), f"{float(v):.1%}"] for i, v in zip(top.indices, top.values)]


# =====================================================================
# 2. Tokenizer
# =====================================================================
def tokenize_text(text):
    if not text or not text.strip():
        raise gr.Error("Please enter a word or sentence.")
    tok, _ = load_gpt2()
    ids = tok(text).input_ids
    rows = [[i, repr(tok.decode(i)), tok.convert_ids_to_tokens(i)] for i in ids]
    return rows, f"{len(ids)} tokens"


# =====================================================================
# 3. Sentiment (general / FinBERT)
# =====================================================================
SENTIMENT_MODELS = {
    "General (DistilBERT SST-2)": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "Financial (FinBERT)": "ProsusAI/finbert",
}


@lru_cache(maxsize=None)
def sentiment_pipe(model_name):
    return pipeline("text-classification", model=model_name)


def analyze_sentiment(text, which):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    preds = sentiment_pipe(SENTIMENT_MODELS[which])(text[:1500], top_k=None)
    return {p["label"]: float(p["score"]) for p in preds}


# =====================================================================
# 4. Named Entity Recognition
# =====================================================================
@lru_cache(maxsize=1)
def ner_pipe():
    return pipeline(
        "token-classification",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
    )


def run_ner(text):
    if not text or not text.strip():
        raise gr.Error("Please enter some text.")
    ents = sorted(ner_pipe()(text), key=lambda e: e["start"])
    out, pos = [], 0
    for e in ents:
        if e["start"] > pos:
            out.append((text[pos:e["start"]], None))
        out.append((text[e["start"]:e["end"]], e["entity_group"]))
        pos = e["end"]
    if pos < len(text):
        out.append((text[pos:], None))
    return out


# =====================================================================
# 5. Question Answering (extractive, DistilBERT-SQuAD)
# =====================================================================
QA_CONTEXT = (
    "Financial sentiment analysis is a challenging task due to the specialized "
    "language and lack of labeled data in that domain. General-purpose models are "
    "not effective enough because of the specialized language used in a financial "
    "context. We hypothesize that pre-trained language models can help with this "
    "problem because they require fewer labeled examples and they can be further "
    "trained on domain-specific corpora. We introduce FinBERT, a language model "
    "based on BERT, to tackle NLP tasks in the financial domain. Our results show "
    "improvement in every measured metric on current state-of-the-art results for "
    "two financial sentiment analysis datasets. We find that even with a smaller "
    "training set and fine-tuning only a part of the model, FinBERT outperforms "
    "state-of-the-art machine learning methods."
)


@lru_cache(maxsize=1)
def load_qa():
    name = "distilbert/distilbert-base-uncased-distilled-squad"
    return AutoTokenizer.from_pretrained(name), AutoModelForQuestionAnswering.from_pretrained(name).eval()


def answer_question(question, context):
    if not question or not question.strip() or not context or not context.strip():
        raise gr.Error("Please enter both a question and a context passage.")
    tok, model = load_qa()
    inputs = tok(question, context, return_tensors="pt", truncation="only_second", max_length=512)
    with torch.no_grad():
        out = model(**inputs)
    start = int(torch.argmax(out.start_logits))
    end = int(torch.argmax(out.end_logits)) + 1
    if end <= start:
        return "(No answer found in the passage)"
    ids = inputs["input_ids"][0][start:end]
    return tok.convert_tokens_to_string(tok.convert_ids_to_tokens(ids))


# =====================================================================
# 6. Machine translation (NLLB-200 distilled 600M)
# =====================================================================
LANGS = {
    "French (fra_Latn)": "fra_Latn",
    "Vietnamese (vie_Latn)": "vie_Latn",
    "Spanish (spa_Latn)": "spa_Latn",
    "German (deu_Latn)": "deu_Latn",
    "Japanese (jpn_Jpan)": "jpn_Jpan",
}


@lru_cache(maxsize=1)
def load_nllb():
    name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(name, src_lang="eng_Latn")
    return tok, AutoModelForSeq2SeqLM.from_pretrained(name).eval()


def translate(text, target):
    if not text or not text.strip():
        raise gr.Error("Please enter an English sentence to translate.")
    tok, model = load_nllb()
    inputs = tok(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[target]),
            max_length=200,
        )
    return tok.decode(out[0], skip_special_tokens=True)


# =====================================================================
# 7. Audio: waveform/mel, classification (AST), speech->text, text->speech
# =====================================================================
def plot_audio(path):
    if not path:
        raise gr.Error("Please upload or record an audio file.")
    y, sr = librosa.load(path)
    fig = Figure(figsize=(10, 7), dpi=100)
    ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(np.arange(len(y)) / sr, y, linewidth=0.5)
    ax1.set_title(f"Waveform  (sr={sr} Hz, {len(y) / sr:.1f}s)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amplitude")
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="log", fmax=8000, ax=ax2)
    ax2.set_title("Mel spectrogram")
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    fig.tight_layout()
    return fig


@lru_cache(maxsize=1)
def load_ast():
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    return AutoFeatureExtractor.from_pretrained(name), ASTForAudioClassification.from_pretrained(name).eval()


def classify_audio(path):
    if not path:
        raise gr.Error("Please upload or record an audio file.")
    fe, model = load_ast()
    y, sr = librosa.load(path, sr=16000)
    inputs = fe(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    probs = torch.sigmoid(logits)  # AudioSet is multi-label
    top = torch.topk(probs, k=5)
    return {model.config.id2label[int(i)]: float(v) for i, v in zip(top.indices, top.values)}


@lru_cache(maxsize=1)
def asr_pipe():
    return pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")


def speech_to_text(path):
    if not path:
        raise gr.Error("Please upload or record an English audio file.")
    y, _ = librosa.load(path, sr=16000)
    return asr_pipe()({"raw": y, "sampling_rate": 16000})["text"]


@lru_cache(maxsize=1)
def tts_pipe():
    return pipeline("text-to-speech", model="suno/bark-small")


def text_to_speech(text):
    if not text or not text.strip():
        raise gr.Error("Please enter an English sentence.")
    out = tts_pipe()(text[:200])
    return out["sampling_rate"], np.asarray(out["audio"]).squeeze()


# =====================================================================
# 8. Image processing (RGB channels, grayscale, resize)
# =====================================================================
def process_image(image, ratio):
    if image is None:
        raise gr.Error("Please upload an image.")
    image = image.convert("RGB")
    arr = np.asarray(image)

    red = arr.copy()
    red[:, :, 1:] = 0
    green = arr.copy()
    green[:, :, [0, 2]] = 0
    blue = arr.copy()
    blue[:, :, :2] = 0

    w, h = image.size
    resized = image.resize((max(1, int(w * ratio)), max(1, int(h * ratio))))
    info = f"Shape: {arr.shape} | Resize: {w}x{h} -> {resized.size[0]}x{resized.size[1]}"
    gallery = [
        (image, "Original"),
        (Image.fromarray(red), "Red channel"),
        (Image.fromarray(green), "Green channel"),
        (Image.fromarray(blue), "Blue channel"),
        (image.convert("L"), "Grayscale"),
        (resized, f"Resize x{ratio}"),
    ]
    return gallery, info


# =============================== UI ===============================
with gr.Blocks(title="Understanding Transformers") as demo:
    gr.Markdown(
        "# 📉 Understanding Transformers\n"
        "A self-study AI project exploring the **Transformer** architecture through "
        "hands-on experiments with Hugging Face models (text, audio, images). "
        "Models are downloaded on first use, so the first run of each tab may take a while."
    )

    # ---- GPT-2 ----
    with gr.Tab("✍️ GPT-2"):
        with gr.Tabs():
            with gr.Tab("Text generation"):
                g_prompt = gr.Textbox(label="Prompt", value="What is machine learning?", lines=2)
                with gr.Row():
                    g_strategy = gr.Dropdown(
                        ["Greedy", "Top-k", "Top-p (Nucleus)", "Temperature", "Random"],
                        value="Top-k", label="Sampling strategy",
                    )
                    g_tokens = gr.Slider(5, 100, value=40, step=5, label="New tokens to generate")
                with gr.Row():
                    g_k = gr.Slider(1, 100, value=50, step=1, label="k (Top-k)")
                    g_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="p (Top-p)")
                    g_temp = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                    g_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                g_out = gr.Textbox(label="Output", lines=8)
                gr.Button("Generate", variant="primary").click(
                    generate, [g_prompt, g_strategy, g_tokens, g_k, g_p, g_temp, g_seed], g_out
                )
            with gr.Tab("Next-token probabilities"):
                n_in = gr.Textbox(
                    label="Sentence",
                    value="I learn machine learning to enhance our understanding of the brain in",
                    lines=2,
                )
                n_out = gr.Dataframe(headers=["Token", "Probability"], interactive=False, label="Top 10")
                gr.Button("Predict next token", variant="primary").click(next_token_probs, n_in, n_out)

    # ---- Tokenizer ----
    with gr.Tab("🔤 Tokenizer"):
        t_in = gr.Textbox(label="Word / sentence", value="antidisestablishmentarianism")
        t_count = gr.Textbox(label="Count", interactive=False)
        t_out = gr.Dataframe(headers=["ID", "Token (decoded)", "Raw token"], interactive=False)
        gr.Button("Tokenize (GPT-2)", variant="primary").click(tokenize_text, t_in, [t_out, t_count])
        gr.Examples(
            [["unbelivable"], ["homoscedasticity"], ["pneumonoultramicroscopicsilicovolcanoconiosis"],
             ["floccinaucinihilipilification"]],
            inputs=t_in, cache_examples=False,
        )

    # ---- Sentiment ----
    with gr.Tab("💬 Sentiment"):
        s_which = gr.Radio(list(SENTIMENT_MODELS), value="General (DistilBERT SST-2)", label="Model")
        s_in = gr.Textbox(label="Text", lines=3, value="This day is great!")
        s_out = gr.Label(label="Result")
        gr.Button("Analyze", variant="primary").click(analyze_sentiment, [s_in, s_which], s_out)
        gr.Examples(
            [["This day is terrible and i am so sad"],
             ["The company reported a strong increase in quarterly revenue, exceeding market expectations."],
             ["Shares fell after the firm reported lower-than-expected earnings"]],
            inputs=s_in, cache_examples=False,
        )

    # ---- NER ----
    with gr.Tab("🏷️ NER"):
        ner_in = gr.Textbox(
            label="Text", lines=3,
            value="Apple announced record earnings in the United States on Monday.",
        )
        ner_out = gr.HighlightedText(label="Entities")
        gr.Button("Recognize entities", variant="primary").click(run_ner, ner_in, ner_out)
        gr.Examples(
            [["I live in UK worked at Facebook after graduating from Harvard"]],
            inputs=ner_in, cache_examples=False,
        )

    # ---- QA ----
    with gr.Tab("❓ Question answering"):
        qa_ctx = gr.Textbox(label="Context passage", value=QA_CONTEXT, lines=8)
        qa_q = gr.Textbox(label="Question", value="What does FinBERT outperform?")
        qa_out = gr.Textbox(label="Answer", interactive=False)
        gr.Button("Answer", variant="primary").click(answer_question, [qa_q, qa_ctx], qa_out)

    # ---- Translation ----
    with gr.Tab("🌍 Translation"):
        tr_in = gr.Textbox(label="English text", value="What is your name?", lines=2)
        tr_lang = gr.Dropdown(list(LANGS), value="Vietnamese (vie_Latn)", label="Translate to")
        tr_out = gr.Textbox(label="Translation", lines=2, interactive=False)
        gr.Button("Translate", variant="primary").click(translate, [tr_in, tr_lang], tr_out)

    # ---- Audio ----
    with gr.Tab("🎧 Audio"):
        with gr.Tabs():
            with gr.Tab("Waveform & spectrogram"):
                a1 = gr.Audio(type="filepath", label="Audio", value=AUDIO_SAMPLE)
                a1_out = gr.Plot()
                gr.Button("Plot", variant="primary").click(plot_audio, a1, a1_out)
            with gr.Tab("Classification (AST)"):
                a2 = gr.Audio(type="filepath", label="Audio", value=AUDIO_SAMPLE)
                a2_out = gr.Label(num_top_classes=5, label="Top 5 AudioSet labels")
                gr.Button("Classify", variant="primary").click(classify_audio, a2, a2_out)
            with gr.Tab("Speech → Text"):
                a3 = gr.Audio(type="filepath", label="English audio", value=AUDIO_SAMPLE)
                a3_out = gr.Textbox(label="Transcript", lines=4, interactive=False)
                gr.Button("Transcribe", variant="primary").click(speech_to_text, a3, a3_out)
            with gr.Tab("Text → Speech"):
                tts_in = gr.Textbox(label="English text (max 200 characters)", value="I like coding in python")
                tts_out = gr.Audio(label="Generated speech")
                gr.Markdown("*Bark-small is quite slow on CPU, please be patient.*")
                gr.Button("Speak", variant="primary").click(text_to_speech, tts_in, tts_out)

    # ---- Image ----
    with gr.Tab("🖼️ Image processing"):
        i_in = gr.Image(type="pil", label="Image")
        i_ratio = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="Resize ratio")
        i_info = gr.Textbox(label="Info", interactive=False)
        i_out = gr.Gallery(label="Results", columns=3, height="auto")
        gr.Button("Process", variant="primary").click(process_image, [i_in, i_ratio], [i_out, i_info])
        if IMAGE_SAMPLES:
            gr.Examples(IMAGE_SAMPLES, inputs=i_in, cache_examples=False)

if __name__ == "__main__":
    demo.queue().launch()
