import html
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import gradio as gr

from src.pipeline import load_model, stream_video, write_to_supabase

C50 = "#f3e2d0"
C100 = "#f0d9c4"
C200 = "#e8c6a8"
C300 = "#ddac85"
C400 = "#cc8a62"
C500 = "#b55937"
C600 = "#a04c2e"
C700 = "#853c25"
C800 = "#6a2d1b"
C900 = "#4b1d12"
C950 = "#2e0e09"
SAGE = "#a3b18a"
PRIMARY_BUTTON = "#c9714d"
PRIMARY_BUTTON_HOVER = "#b55937"
BACKGROUND = C50

HISTORY_KEY = "oreo_analysis_history"
REPORTS_DIR = Path(tempfile.gettempdir()) / "oreo_reports"
TITLE = "Oreo Visibility Analysis"
SUBTITLE = "Automated detection and reporting of brand appearances in video content"
URL_PLACEHOLDER = "Paste a video URL"
UPLOAD_LABEL = "Upload video"
ANALYSE_LABEL = "Analyse"
DOWNLOAD_LABEL = "Download Report"
HISTORY_IMAGES_NOTE = "Images available in database only"

OREO_PRIMARY_HUE = gr.themes.Color(
    c50=C50,
    c100=C100,
    c200=C200,
    c300=C300,
    c400=C400,
    c500=C500,
    c600=C600,
    c700=C700,
    c800=C800,
    c900=C900,
    c950=C950,
    name="oreo",
)

OREO_THEME = gr.themes.Base(
    primary_hue=OREO_PRIMARY_HUE,
    neutral_hue="stone",
    font=[
        gr.themes.GoogleFont("Playfair Display"),
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Inter"),
        "ui-monospace",
        "monospace",
    ],
).set(
    body_background_fill=BACKGROUND,
    button_primary_background_fill=PRIMARY_BUTTON,
    button_primary_background_fill_hover=PRIMARY_BUTTON_HOVER,
    color_accent=C500,
)

CSS = f"""
.gradio-container {{
    font-family: "Inter", ui-sans-serif, sans-serif;
}}
.oreo-title h1, .oreo-title {{
    font-family: "Playfair Display", serif;
    color: {C950};
    margin-bottom: 0;
}}
.oreo-subtitle {{
    color: {C700};
    margin-top: 4px;
    margin-bottom: 24px;
}}
.oreo-results {{
    max-height: 58vh;
    overflow-y: auto;
    padding-right: 8px;
}}
.oreo-card {{
    background-color: {SAGE};
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 12px;
    align-items: center;
    gap: 20px;
}}
.oreo-card-meta h4 {{
    font-family: "Playfair Display", serif;
    color: {C950};
    font-size: 1.25rem;
    margin: 0 0 8px 0;
}}
.oreo-card-meta p {{
    color: {C950};
    margin: 2px 0;
    font-size: 0.92rem;
}}
.oreo-summary {{
    background-color: #f7ece0;
    border: 1px solid {C500};
    border-radius: 16px;
    padding: 16px;
    margin-top: 8px;
}}
.oreo-summary h3 {{
    font-family: "Playfair Display", serif;
    color: {C950};
    margin-top: 0;
}}
.oreo-summary table {{
    width: 100%;
    border-collapse: collapse;
}}
.oreo-summary th, .oreo-summary td {{
    text-align: left;
    padding: 8px;
    border-bottom: 1px solid {C300};
    color: {C950};
}}
.oreo-warning {{
    color: {C700};
    font-size: 0.85rem;
    margin-top: 6px;
}}
.oreo-history-item {{
    background-color: #f7ece0;
    border: 1px solid {C400};
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 10px;
}}
.oreo-history-item summary {{
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    color: {C950};
}}
.oreo-history-name {{
    font-weight: 600;
}}
.oreo-history-time {{
    color: {C700};
    font-size: 0.82rem;
}}
.oreo-history-report {{
    white-space: pre-wrap;
    font-family: "Inter", ui-monospace, monospace;
    font-size: 0.8rem;
    color: {C950};
    margin-top: 10px;
}}
.oreo-history-note {{
    color: {C700};
    font-size: 0.75rem;
    margin-top: 8px;
}}
.oreo-history-empty {{
    color: {C700};
}}
.oreo-clear-btn {{
    background-color: {PRIMARY_BUTTON};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    cursor: pointer;
    margin-top: 8px;
}}
.oreo-clear-btn:hover {{
    background-color: {PRIMARY_BUTTON_HOVER};
}}
"""

HISTORY_TEMPLATE = """
<div id="oreo-history-list">
    <p class="oreo-history-empty">No analyses yet</p>
</div>
<button id="oreo-history-clear" class="oreo-clear-btn" type="button">Clear history</button>
"""

HISTORY_JS = f"""
const KEY = '{HISTORY_KEY}';
const root = element.querySelector('#oreo-history-list');
const clearBtn = element.querySelector('#oreo-history-clear');

function readHistory() {{
    try {{
        const raw = window.localStorage.getItem(KEY);
        return raw ? JSON.parse(raw) : [];
    }} catch (err) {{
        return [];
    }}
}}

function escapeHtml(value) {{
    return String(value == null ? '' : value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}}

function renderHistory() {{
    const items = readHistory();
    if (!items.length) {{
        root.innerHTML = '<p class="oreo-history-empty">No analyses yet</p>';
        return;
    }}
    let markup = '';
    for (let i = 0; i < items.length; i += 1) {{
        const item = items[i];
        const name = escapeHtml(item.video_name || item.url_or_filename || 'Untitled');
        const time = escapeHtml(item.processed_at || '');
        markup += '<details class="oreo-history-item">';
        markup += '<summary><span class="oreo-history-name">' + name + '</span>';
        markup += '<span class="oreo-history-time">' + time + '</span></summary>';
        markup += '<pre class="oreo-history-report">' + escapeHtml(item.report_text || '') + '</pre>';
        markup += '<p class="oreo-history-note">{HISTORY_IMAGES_NOTE}</p>';
        markup += '</details>';
    }}
    root.innerHTML = markup;
}}

function saveEntry() {{
    const raw = typeof props.value === 'string' ? props.value : JSON.stringify(props.value);
    if (!raw) {{
        return;
    }}
    let entry;
    try {{
        entry = JSON.parse(raw);
    }} catch (err) {{
        return;
    }}
    const items = readHistory();
    items.unshift(entry);
    window.localStorage.setItem(KEY, JSON.stringify(items));
    renderHistory();
}}

renderHistory();
if (clearBtn) {{
    clearBtn.addEventListener('click', function () {{
        window.localStorage.removeItem(KEY);
        renderHistory();
    }});
}}
if (typeof watch === 'function') {{
    watch('value', saveEntry);
}}
"""

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def format_timestamp(seconds: float) -> str:
    minutes, remaining = divmod(max(float(seconds), 0.0), 60)
    return f"{int(minutes):02d}:{remaining:05.2f}"


def render_summary(report: dict) -> str:
    rows = []
    for class_name, class_report in report["classes"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            f"<td>{class_report['total_visible_seconds']:.2f} s</td>"
            f"<td>{class_report['percentage']:.2f}%</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='3'>No brand appearances detected</td></tr>")
    return (
        "<div class='oreo-summary'>"
        "<h3>Summary</h3>"
        f"<p>Video: {html.escape(report['video_name'])}</p>"
        f"<p>Duration: {report['duration_seconds']:.2f} s</p>"
        "<table><thead><tr><th>Class</th><th>Visible time</th>"
        "<th>Percentage</th></tr></thead><tbody>"
        f"{''.join(rows)}"
        "</tbody></table>"
        "</div>"
    )


def report_to_markdown(report: dict, processed_at: str) -> str:
    lines = [
        f"# Oreo Visibility Report: {report['video_name']}",
        "",
        f"Processed at: {processed_at}",
        f"Duration: {report['duration_seconds']:.2f} s",
        f"FPS: {report['fps']:.2f}",
        f"Total frames: {report['total_frames']}",
        "",
        "## Class Totals",
        "",
        "| Class | Visible seconds | Percentage |",
        "| --- | --- | --- |",
    ]
    for class_name, class_report in report["classes"].items():
        lines.append(
            f"| {class_name} | {class_report['total_visible_seconds']:.2f} "
            f"| {class_report['percentage']:.2f}% |"
        )
    if not report["classes"]:
        lines.append("| None | 0.00 | 0.00% |")

    lines.extend(["", "## Intervals", ""])
    for class_name, class_report in report["classes"].items():
        lines.append(f"### {class_name}")
        lines.append("")
        for interval in class_report["intervals"]:
            lines.append(
                f"Start: {format_timestamp(interval['start_time'])} "
                f"to {format_timestamp(interval['end_time'])}"
            )
            lines.append(f"Duration: {interval['duration']:.2f} s")
            lines.append(f"Peak confidence: {interval['peak_confidence'] * 100:.2f}%")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report_file(report: dict, report_text: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{report['video_name']}_report.md"
    path.write_text(report_text, encoding="utf-8")
    return path


def resolve_source(url_value: str, file_value: str | None) -> tuple[str, str]:
    url = (url_value or "").strip()
    if url:
        return url, url
    if file_value:
        path = file_value if isinstance(file_value, str) else file_value[0]
        return path, Path(path).name
    raise ValueError("Provide a video URL or upload a video")


def sync_inputs(url_value: str, file_value: str | None):
    has_url = bool((url_value or "").strip())
    has_file = bool(file_value)
    upload_label = UPLOAD_LABEL
    if has_file and isinstance(file_value, str):
        upload_label = f"Video: {Path(file_value).name}"
    return (
        gr.update(interactive=not has_url, label=upload_label),
        gr.update(interactive=not has_file),
        gr.update(interactive=has_url or has_file),
    )


def analyse(url_value: str, file_value: str | None):
    try:
        source, label = resolve_source(url_value, file_value)
    except ValueError as exc:
        yield {"status": gr.update(value=str(exc))}
        return

    entries: list[dict] = []
    report = None
    yield {
        "analyse_button": gr.update(interactive=False),
        "status": gr.update(value="Analysing video"),
        "intervals_store": [],
        "summary": gr.update(value="", visible=False),
        "download_button": gr.update(value=None, visible=False),
        "warning": gr.update(visible=False),
    }
    try:
        model = get_model()
        for event in stream_video(source, model=model):
            if event["type"] == "interval":
                entries.append(event["data"])
                yield {
                    "intervals_store": list(entries),
                    "status": gr.update(
                        value=f"Found {len(entries)} brand appearances"
                    ),
                }
            elif event["type"] == "report":
                report = event["data"]
    except Exception as exc:  # noqa: BLE001
        yield {
            "analyse_button": gr.update(interactive=True),
            "status": gr.update(value=f"Analysis failed: {exc}"),
        }
        return

    if report is None:
        yield {
            "analyse_button": gr.update(interactive=True),
            "status": gr.update(value="Analysis finished without a report"),
        }
        return

    processed_at = datetime.now(UTC).isoformat(timespec="seconds")
    report_text = report_to_markdown(report, processed_at)
    report_path = write_report_file(report, report_text)
    history_entry = {
        "video_name": report["video_name"],
        "url_or_filename": label,
        "processed_at": processed_at,
        "report_text": report_text,
    }
    yield {
        "intervals_store": list(entries),
        "status": gr.update(
            value=f"Analysis complete. {len(entries)} brand appearances found"
        ),
        "summary": gr.update(value=render_summary(report), visible=True),
        "download_button": gr.update(value=str(report_path), visible=True),
        "analyse_button": gr.update(interactive=True),
        "history_out": json.dumps(history_entry),
        "report_state": report,
        "warning": gr.update(visible=False),
    }


def persist_report(report: dict | None):
    if not report:
        return gr.update(visible=False)
    try:
        write_to_supabase(report)
    except Exception as exc:  # noqa: BLE001
        message = html.escape(str(exc))
        return gr.update(
            value=f"<p class='oreo-warning'>Database save failed: {message}</p>",
            visible=True,
        )
    return gr.update(visible=False)


def render_card(interval: dict):
    class_name = html.escape(interval["class_name"])
    start = format_timestamp(interval["start_time"])
    end = format_timestamp(interval["end_time"])
    duration = interval["duration"]
    confidence = interval["peak_confidence"] * 100
    with gr.Row(elem_classes=["oreo-card"], key=interval["interval_id"]):
        gr.HTML(
            "<div class='oreo-card-meta'>"
            f"<h4>{class_name}</h4>"
            f"<p>{start} &rarr; {end}</p>"
            f"<p>Duration: {duration:.2f} s</p>"
            f"<p>Peak confidence: {confidence:.1f}%</p>"
            "</div>"
        )
        gr.Image(
            value=interval["crop_path"],
            show_label=False,
            container=False,
            interactive=False,
            width=200,
            height=150,
            key=f"image_{interval['interval_id']}",
        )


def build_demo() -> gr.Blocks:
    history_template = HISTORY_TEMPLATE
    history_js = HISTORY_JS

    with gr.Blocks(
        theme=OREO_THEME,
        css=CSS,
        title=TITLE,
    ) as demo:
        gr.HTML(f"<h1 class='oreo-title'>{TITLE}</h1>")
        gr.HTML(f"<p class='oreo-subtitle'>{SUBTITLE}</p>")

        with gr.Tabs():
            with gr.Tab("Analysis"):
                with gr.Row():
                    url_input = gr.Textbox(
                        label="Video URL",
                        placeholder=URL_PLACEHOLDER,
                        lines=2,
                        scale=3,
                    )
                    upload_input = gr.UploadButton(
                        UPLOAD_LABEL,
                        file_count="single",
                        file_types=["video"],
                        scale=1,
                    )
                analyse_button = gr.Button(
                    ANALYSE_LABEL,
                    variant="primary",
                    interactive=False,
                )
                status = gr.Markdown("")
                warning = gr.HTML(visible=False)

                intervals_store = gr.JSON(value=[], visible="hidden")

                with gr.Column(elem_classes=["oreo-results"]):

                    @gr.render(inputs=intervals_store)
                    def render_results(intervals):
                        if not intervals:
                            return
                        for interval in intervals:
                            render_card(interval)

                summary = gr.HTML(visible=False)
                download_button = gr.DownloadButton(
                    DOWNLOAD_LABEL,
                    visible=False,
                )

                report_state = gr.State(None)

            with gr.Tab("History"):
                history_out = gr.HTML(
                    value="",
                    html_template=history_template,
                    js_on_load=history_js,
                )

        url_input.change(
            sync_inputs,
            inputs=[url_input, upload_input],
            outputs=[upload_input, url_input, analyse_button],
        )
        upload_input.change(
            sync_inputs,
            inputs=[url_input, upload_input],
            outputs=[upload_input, url_input, analyse_button],
        )

        analysis_event = analyse_button.click(
            analyse,
            inputs=[url_input, upload_input],
            outputs=[
                intervals_store,
                status,
                summary,
                download_button,
                analyse_button,
                history_out,
                report_state,
                warning,
            ],
        )
        analysis_event.then(
            persist_report,
            inputs=[report_state],
            outputs=[warning],
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    _MODEL = load_model()
    demo.launch(share=True, server_name="0.0.0.0")
