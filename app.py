"""
app.py — Gradio web interface for GeoIdentifier
Run: python app.py
"""

import logging
import tempfile

import gradio as gr
from src.identifier import GeoIdentifier, IdentificationResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

identifier = GeoIdentifier()


def build_map_url(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"


def result_to_markdown(result: IdentificationResult) -> str:
    if not result.predictions:
        return "Could not determine location from this image."

    lines = [f"**Strategy used:** `{result.strategy_used}`\n"]
    for i, pred in enumerate(result.predictions[:6]):
        icon = "📍" if pred.source == "exif" else "🧠"
        lines.append(f"### {icon} #{i+1} — {pred.source.upper()}")
        lines.append(f"**Address:** {pred.address}")
        lines.append(f"**Coordinates:** `{pred.lat:.5f}, {pred.lon:.5f}`")
        lines.append(f"**Confidence:** {pred.confidence * 100:.1f}%")
        lines.append(f"[Open in Google Maps]({build_map_url(pred.lat, pred.lon)})\n")
    return "\n".join(lines)


def trace_to_markdown(result: IdentificationResult) -> str:
    if not result.process_trace:
        return "No process details available."

    lines = ["### How the model narrowed this down"]
    for i, step in enumerate(result.process_trace, start=1):
        lines.append(f"{i}. {step}")
    return "\n".join(lines)


def identify(image):
    if image is None:
        return "Please upload an image.", "", ""

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, format="JPEG")
        result = identifier.identify(tmp.name)

    md = result_to_markdown(result)
    trace_md = trace_to_markdown(result)
    map_html = ""
    if result.best:
        lat, lon = result.best.lat, result.best.lon
        map_html = (
            f'<iframe width="100%" height="300" style="border:none;border-radius:12px" '
            f'src="https://maps.google.com/maps?q={lat},{lon}&z=13&output=embed"></iframe>'
        )
    return md, map_html, trace_md


with gr.Blocks(
    title="GeoIdentifier",
    theme=gr.themes.Base(
        primary_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
) as demo:
    gr.Markdown("# GeoIdentifier\n### Location identification — EXIF + GeoCLIP/CLIP Ensemble\n---")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            run_btn = gr.Button("Identify Location", variant="primary")
        with gr.Column(scale=2):
            result_md = gr.Markdown(label="Results")
            map_output = gr.HTML(label="Map")
            trace_output = gr.Markdown(label="How It Narrowed Down")

    run_btn.click(
        fn=identify,
        inputs=[image_input],
        outputs=[result_md, map_output, trace_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)
