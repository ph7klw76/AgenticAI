# Getting up Gemma4
You need to open a huggingface account

To get your token:

Log into Hugging Face.
Go to your account settings.
Create a User Access Token.
Paste it when hf auth login asks

```text

module load miniconda/24.1.2

export HF_HOME=/scr/user/woon/.cache/huggingface
export HF_HUB_CACHE=/scr/user/woon/.cache/huggingface/hub
export HF_TOKEN_PATH=/home/user/woon/.cache/huggingface/token

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" /home/user/woon/.cache/huggingface
conda create -n gemma4 python=3.11 -y
conda activate gemma4

pip install -U pip
pip install -U transformers==5.50 torch==2.60 accelerate==1.130 huggingface_hub==1.9.0 torchvision==0.21.0 torchaudio==2.6.0 PIL: 12.2.0 PIL: 12.2.0 pillow gradio

hf auth login
```text


example of code to to the chat:



```python
import os
import argparse
from pathlib import Path
from datetime import datetime
from textwrap import wrap

from transformers import pipeline
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def save_text(text: str, path: Path) -> None:
    path.write_text(text, encoding="utf-8")


def save_pdf(text: str, path: Path, title: str) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    left_margin = 50
    right_margin = 50
    top_margin = 50
    bottom_margin = 50
    line_height = 14
    usable_width = width - left_margin - right_margin

    # Built-in Helvetica is widely available. For heavy Unicode use, register a TTF font.
    title_font = "Helvetica-Bold"
    body_font = "Helvetica"
    body_size = 11

    c.setTitle(title)
    y = height - top_margin

    c.setFont(title_font, 14)
    c.drawString(left_margin, y, title)
    y -= 24

    c.setFont(body_font, body_size)

    # Rough character width estimate for wrapping plain text.
    max_chars = max(40, int(usable_width / 6.2))

    for paragraph in text.splitlines() or [text]:
        lines = wrap(paragraph, width=max_chars) if paragraph.strip() else [""]
        for line in lines:
            if y < bottom_margin:
                c.showPage()
                y = height - top_margin
                c.setFont(body_font, body_size)
            c.drawString(left_margin, y, line)
            y -= line_height
        y -= 4

    c.save()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4, print the full answer, and save it to TXT + PDF."
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("MODEL_ID", "google/gemma-4-31B-it"),
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--prompt",
        default="Explain superconductivity from a quantum mechanics perspective with equations.",
        help="User prompt",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant. Answer clearly and completely.",
        help="System prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "2048")),
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.7")),
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("TOP_P", "0.95")),
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "."),
        help="Directory to save TXT and PDF outputs",
    )
    parser.add_argument(
        "--output-prefix",
        default=os.getenv("OUTPUT_PREFIX", "gemma4_response"),
        help="Base filename without extension",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable sampling and use greedy decoding",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = pipeline(
        task="any-to-any",
        model=args.model_id,
        device_map="auto",
        dtype="auto",
    )

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": args.system}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": args.prompt}],
        },
    ]

    # Pass generation parameters explicitly. Do not also pass generation_config.
    result = pipe(
        messages,
        return_full_text=False,
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sampling,
        temperature=args.temperature if not args.no_sampling else None,
        top_p=args.top_p if not args.no_sampling else None,
    )

    response_text = result[0]["generated_text"].strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = output_dir / f"{args.output_prefix}_{timestamp}.txt"
    pdf_path = output_dir / f"{args.output_prefix}_{timestamp}.pdf"

    save_text(response_text, txt_path)
    save_pdf(response_text, pdf_path, title=f"Gemma Output - {args.model_id}")

    print("\n=== GEMMA RESPONSE START ===\n")
    print(response_text)
    print("\n=== GEMMA RESPONSE END ===\n")
    print(f"Saved text: {txt_path}")
    print(f"Saved PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
```


```text
module load miniconda/24.1.2
source ~/.bashrc

conda activate gemma4

export HF_HOME=/scr/user/woon/.cache/huggingface
export HF_HUB_CACHE=/scr/user/woon/.cache/huggingface/hub
export HF_TOKEN_PATH=/home/user/woon/.cache/huggingface/token


unset PYTHONPATH
export PYTHONNOUSERSITE=1

python test_gemma4.py
111

