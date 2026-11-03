import base64
import json
import tempfile
from datetime import UTC, datetime

import gradio as gr

from src.pipeline import load_model, stream_video, write_to_supabase

_model = load_model()

_PRIMARY = gr.themes.Color(
    c50="#f3e2d0",
    c100="#e8c9b0",
    c200="#ddb090",
    c300="#d29770",
    c400="#c9714d",
    c500="#b55937",
    c600="#9a4a2e",
    c700="#7f3b25",
    c800="#642c1c",
    c900="#491d13",
    c950="#2e0e09",
)

_NEUTRAL = gr.themes.Color(
    c50="#f1f5ee",
    c100="#e3ebe0",
    c200="#c7d7c1",
    c300="#abc3a2",
    c400="#a3b18a",
    c500="#8a9973",
    c600="#71805c",
    c700="#586745",
    c800="#3f4e2e",
    c900="#2a3520",
    c950="#511407",
)

theme = gr.themes.Base(
    primary_hue=_PRIMARY,
    neutral_hue=_NEUTRAL,
    font=[
        gr.themes.GoogleFont("Inter"),
        gr.themes.GoogleFont("Playfair Display"),
        "ui-sans-serif",
        "sans-serif",
    ],
    font_mono=[gr.themes.GoogleFont("Inter"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#f3e2d0",
    body_background_fill_dark="#f3e2d0",
    button_primary_background_fill="#c9714d",
    button_primary_background_fill_hover="#b55937",
    color_accent="#b55937",
    body_text_color="#b55937",
    body_text_color_dark="#b55937",
    block_label_text_color="#b55937",
    block_label_text_color_dark="#b55937",
    button_large_text_weight="400",
    button_large_text_size="14px",
    button_medium_text_weight="400",
    button_medium_text_size="12px",
)

CSS = (
    "body, .gradio-container { background: #f3e2d0 !important; } "
    ".gradio-container { position: relative !important; } "
    "#app-title, #app-title h1 { font-family: 'Playfair Display', serif !important; "
    "color: #9a4a2e !important; } "
    "#logo-block { position: static !important; padding: 0 !important; "
    "margin: 0 !important; background: transparent !important; "
    "border: none !important; } "
    ".flower-logo { position: absolute; top: 16px; right: -20px; width: 146px; "
    "height: auto; z-index: 1000; pointer-events: none; } "
    "#url-card, #url-card .block { "
    "background: #586745 !important; border-color: #586745 !important; } "
    "#url-card textarea, #url-card input { "
    "background: #a3b18a !important; color: #ffffff !important; "
    "border-color: #a3b18a !important; } "
    "#url-card label, #url-card .block-label, #url-card label span { "
    "color: #ffffff !important; } "
    "#url-card textarea::placeholder { color: #ffffff !important; } "
    "#upload-btn { display: flex !important; justify-content: center !important; } "
    "#upload-btn, #upload-btn button { background: #586745 !important; "
    "color: #ffffff !important; border-color: #586745 !important; "
    "justify-content: center !important; } "
    "#results-scroll { "
    "background: transparent !important; "
    "border: 2px solid #a3b18a; "
    "border-radius: 12px; "
    "padding: 16px 20px; "
    "margin-top: 8px; "
    "min-height: calc(100vh - 360px); "
    "overflow-y: auto; "
    "color: #3f4e2e; "
    "} "
    "#results-scroll > .block, #results-scroll .html-container { "
    "background: transparent !important; border: none !important; }"
)

LOGO_PATH = "public/flower-logo.png"

_JS_FORCE_LIGHT = """
() => {
    const fix = () => {
        document.body.classList.remove('dark');
        document.documentElement.classList.remove('dark');
    };
    fix();
    new MutationObserver(fix).observe(document.body, {
        attributes: true,
        attributeFilter: ['class'],
    });
}
"""

SAGE = "#a3b18a"
CARD = (
    f"background:{SAGE};border-radius:10px;padding:16px;margin-bottom:12px;"
    "display:flex;gap:16px;align-items:flex-start;"
)

_JS_WRITE = """
(payload) => {
    if (!payload) return '';
    try {
        const h = JSON.parse(localStorage.getItem('oreo_history') || '[]');
        h.unshift(JSON.parse(payload));
        localStorage.setItem('oreo_history', JSON.stringify(h.slice(0, 20)));
    } catch(e) {}
    return '';
}
"""

_JS_READ = """
() => {
    try {
        const h = JSON.parse(localStorage.getItem('oreo_history') || '[]');
        if (!h.length) return '<p>No analyses yet.</p>';
        return h.map(e => `
            <details style="margin-bottom:10px;border:1px solid #c9714d;border-radius:6px;padding:10px;">
                <summary style="cursor:pointer;font-weight:600;">
                    ${e.video_name} — ${e.processed_at}
                </summary>
                <pre style="white-space:pre-wrap;font-size:0.82em;margin-top:8px;">${e.report_text}</pre>
                <p style="font-size:0.78em;color:#7f3b25;font-style:italic;">
                    Images available in database only.
                </p>
            </details>
        `).join('');
    } catch(e) {
        return '<p>Could not read history.</p>';
    }
}
"""

_JS_CLEAR = """
() => {
    localStorage.removeItem('oreo_history');
    return '<p>History cleared.</p>';
}
"""


def _b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _card(iv: dict) -> str:
    class_name = iv["interval_id"].rsplit("_", 1)[0]
    b64 = _b64(iv.get("crop_path", ""))
    img = (
        f'<img src="data:image/jpeg;base64,{b64}" '
        'style="width:120px;height:80px;object-fit:cover;border-radius:6px;flex-shrink:0;">'
        if b64
        else ""
    )
    return (
        f'<div style="{CARD}">{img}'
        f"<div><strong>{class_name}</strong><br>"
        f"{iv['start_time']:.2f}s → {iv['end_time']:.2f}s &nbsp;|&nbsp; "
        f"Duration: {iv['duration']:.2f}s &nbsp;|&nbsp; "
        f"Confidence: {iv['peak_confidence']:.0%}</div></div>"
    )


def _summary(report: dict) -> str:
    rows = "".join(
        f"<p><strong>{cls}:</strong> {d['total_visible_seconds']}s ({d['percentage']}%)</p>"
        for cls, d in report["classes"].items()
    )
    return f'<div style="margin-top:20px;"><h3>Summary</h3>{rows}</div>'


def _md(report: dict) -> str:
    lines = [
        f"# Oreo Visibility Report — {report['video_name']}",
        "",
        f"**Duration:** {report['duration_seconds']:.2f}s | "
        f"**FPS:** {report['fps']} | **Frames:** {report['total_frames']}",
        "",
    ]
    for cls, data in report["classes"].items():
        lines += [
            f"## {cls}",
            f"**Total visible:** {data['total_visible_seconds']}s ({data['percentage']}%)",
            "",
            "| # | Start | End | Duration | Confidence |",
            "|---|-------|-----|----------|------------|",
        ]
        for i, iv in enumerate(data["intervals"]):
            lines.append(
                f"| {i + 1} | {iv['start_time']:.2f}s | {iv['end_time']:.2f}s "
                f"| {iv['duration']:.2f}s | {iv['peak_confidence']:.0%} |"
            )
        lines.append("")
    return "\n".join(lines)


def analyse(url: str, video_path: str | None):
    source = (url or "").strip() or video_path
    if not source:
        yield (
            gr.update(value="<p>Provide a URL or upload a video.</p>"),
            gr.update(visible=False),
            "",
        )
        return

    cards = ""
    report = None

    for event in stream_video(source, model=_model):
        if event["type"] == "interval":
            cards += _card(event["data"])
            yield gr.update(value=cards), gr.update(visible=False), ""
        elif event["type"] == "report":
            report = event["data"]

    if not report:
        yield (
            gr.update(value=cards or "<p>No detections found.</p>"),
            gr.update(visible=False),
            "",
        )
        return

    md_text = _md(report)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    tmp.write(md_text)
    tmp.close()

    warning = ""
    try:
        write_to_supabase(report)
    except Exception as e:
        warning = f'<p style="color:#9a4a2e;font-size:0.85em;">⚠ Supabase write failed: {e}</p>'

    history_payload = json.dumps(
        {
            "video_name": report["video_name"],
            "processed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "report_text": md_text,
        }
    )

    yield (
        gr.update(value=cards + _summary(report) + warning),
        gr.update(visible=True, value=tmp.name),
        history_payload,
    )


with gr.Blocks(title="Oreo Visibility Analysis") as demo:
    gr.HTML(
        elem_id="logo-block",
        value=f'<img class="flower-logo" '
        f'src="data:image/png;base64,{_b64(LOGO_PATH)}" alt="Flower logo">',
    )
    gr.Markdown("# Oreo Visibility Analysis", elem_id="app-title")
    gr.Markdown(
        "Automated detection and reporting of brand appearances in video content",
        elem_id="app-subtitle",
    )

    with gr.Tabs():
        with gr.Tab("Analysis"):
            video_state = gr.State(None)

            with gr.Row():
                url_input = gr.Textbox(
                    placeholder="Paste a video URL",
                    label="Video URL",
                    scale=3,
                    elem_id="url-card",
                )
                upload_btn = gr.UploadButton(
                    "Upload video",
                    file_types=["video"],
                    scale=1,
                    elem_id="upload-btn",
                )

            analyse_btn = gr.Button("Analyse", variant="primary")

            with gr.Column(elem_id="results-scroll"):
                results_html = gr.HTML()

            download_file = gr.File(label="Download report (.md)", visible=False)
            history_trigger = gr.Textbox(visible=False, value="")

            url_input.change(
                lambda url: gr.update(interactive=not bool((url or "").strip())),
                inputs=url_input,
                outputs=upload_btn,
            )
            upload_btn.upload(
                lambda p: (p, gr.update(interactive=False)),
                inputs=upload_btn,
                outputs=[video_state, url_input],
            )
            analyse_btn.click(
                analyse,
                inputs=[url_input, video_state],
                outputs=[results_html, download_file, history_trigger],
            )
            history_trigger.change(
                fn=None,
                inputs=history_trigger,
                outputs=history_trigger,
                js=_JS_WRITE,
            )

        with gr.Tab("History"):
            gr.Markdown("Past analyses — stored locally in your browser.")
            history_html = gr.HTML(value="<p>Click Refresh to load history.</p>")
            with gr.Row():
                refresh_btn = gr.Button("Refresh")
                clear_btn = gr.Button("Clear history", variant="secondary")

            refresh_btn.click(fn=None, outputs=history_html, js=_JS_READ)
            clear_btn.click(fn=None, outputs=history_html, js=_JS_CLEAR)
            demo.load(fn=None, outputs=history_html, js=_JS_READ)
    demo.load(fn=None, js=_JS_FORCE_LIGHT)

if __name__ == "__main__":
    demo.launch(theme=theme, css=CSS, share=True, server_name="0.0.0.0")
