"""
web_app.py  —  Complaint Auto-Routing System · Gradio Web Interface
────────────────────────────────────────────────────────────────

Run:
    python app/web_app.py
    # → opens at http://localhost:7860

Install Gradio:
    pip install gradio

No external API keys required.
"""

import os
import sys
import json
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.engine import ComplaintRoutingEngine, SAVE_DIR

# ─── Automatic Offline Model Training & Setup ───────────────────
def ensure_models_trained():
    required_files = [
        "embedding_engine.pkl",
        "officer_classifier.pkl",
        "priority_classifier.pkl",
        "eta_regressor.pkl",
        "label_encoders.pkl",
        "vector_store.pkl"
    ]
    all_exist = all(os.path.exists(os.path.join(SAVE_DIR, f)) for f in required_files)
    if not all_exist:
        print("[Startup] Missing trained models. Initiating automatic data generation and training...")
        
        # 1. Generate data if missing
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "synthetic_complaints.csv")
        if not os.path.exists(data_path):
            print("[Startup] Generating synthetic complaints dataset...")
            from data.generate_data import generate_complaints
            df = generate_complaints(n_per_officer=100)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"[Startup] Generated {len(df)} complaints.")
            
        # 2. Train models
        print("[Startup] Training models offline...")
        from models.train import main as train_main
        train_main()
        print("[Startup] Model training complete.")

ensure_models_trained()
engine = ComplaintRoutingEngine().load(SAVE_DIR)

# Priority colours (HTML)
PRIORITY_BADGE = {
    "High":   '<span style="display:inline-block;white-space:nowrap;min-width:70px;text-align:center;background:#ef4444;color:#fff;padding:4px 10px;border-radius:4px;font-weight:700;font-size:12px">High</span>',
    "Medium": '<span style="display:inline-block;white-space:nowrap;min-width:70px;text-align:center;background:#f59e0b;color:#fff;padding:4px 10px;border-radius:4px;font-weight:700;font-size:12px">Med</span>',
    "Low":    '<span style="display:inline-block;white-space:nowrap;min-width:70px;text-align:center;background:#22c55e;color:#fff;padding:4px 10px;border-radius:4px;font-weight:700;font-size:12px">Low</span>',
}

DEPT_ICON = {
    "Infrastructure & Roads":    "🛣️",
    "Water & Sanitation":        "💧",
    "Electricity & Utilities":   "⚡",
    "Public Safety & Security":  "🛡️",
    "Health & Environment":      "🌿",
    "Land & Property":           "🏠",
    "Transport & Traffic":       "🚌",
    "Administrative Services":   "📋",
}


def build_output_html(result: dict) -> str:
    o   = result["officer"]
    p   = result["priority"]
    eta = result["eta_days"]
    sim = result.get("similar_complaints", [])
    icon = DEPT_ICON.get(o["department"], "🏛️")
    badge = PRIORITY_BADGE.get(p["level"], p["level"])

    # ── Similar complaints table
    sim_rows = ""
    for s in sim:
        snip = textwrap.shorten(s["text_snippet"].replace("…", ""), width=80)
        sb   = PRIORITY_BADGE.get(s["priority"], s["priority"])
        sim_rows += f"""
        <tr>
          <td style="padding:8px 8px;font-size:12px;color:#475569">{s['complaint_id']}</td>
          <td style="padding:8px 8px;font-size:12px;color:#1e293b;line-height:1.4">{snip}</td>
          <td style="padding:8px 8px;text-align:center">{sb}</td>
          <td style="padding:8px 8px;text-align:center;font-size:12px;color:#1e293b;font-weight:600">{s['eta_days']}d</td>
          <td style="padding:8px 8px;text-align:center;font-size:12px;color:#6366f1;font-weight:600">{s['similarity_score']:.3f}</td>
        </tr>"""

    html = f"""
<div style="font-family:Inter,system-ui,sans-serif;max-width:700px">

  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">

    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:14px">
      <div style="font-size:11px;color:#0369a1;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Assigned Officer</div>
      <div style="font-size:20px;margin:4px 0"></div>
      <div style="font-weight:700;font-size:15px;color:#0c4a6e">{o['name']}</div>
      <div style="font-size:12px;color:#0369a1;margin-top:2px">{o['department']}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px">{o['id']}  ·  {o['confidence']}% conf.</div>
    </div>

    <div style="background:#fefce8;border:1px solid #fde68a;border-radius:10px;padding:14px">
      <div style="font-size:11px;color:#92400e;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Priority</div>
      <div style="font-size:20px;margin:4px 0"></div>
      <div style="margin-top:4px">{badge}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:6px">{p['confidence']}% confidence</div>
    </div>

    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px">
      <div style="font-size:11px;color:#166534;font-weight:600;text-transform:uppercase;letter-spacing:.5px">Est. Resolution</div>
      <div style="font-size:20px;margin:4px 0"></div>
      <div style="font-weight:700;font-size:22px;color:#14532d">{eta}</div>
      <div style="font-size:12px;color:#166534">day(s)</div>
    </div>
  </div>

  <div style="background:#fafafa;border:1px solid #e2e8f0;border-radius:10px;padding:14px">
    <div style="font-size:12px;font-weight:700;color:#1e293b;margin-bottom:8px">Similar Past Complaints (Top {len(sim)})</div>
    <table style="width:100%;border-collapse:collapse">
      <thead>
        <tr style="border-bottom:1px solid #e2e8f0">
          <th style="text-align:left;font-size:11px;color:#475569;padding:6px 8px;width:70px">ID</th>
          <th style="text-align:left;font-size:11px;color:#475569;padding:6px 8px">Snippet</th>
          <th style="font-size:11px;color:#475569;padding:6px 8px;width:105px;text-align:center">Priority</th>
          <th style="font-size:11px;color:#475569;padding:6px 8px;width:60px;text-align:center">ETA</th>
          <th style="font-size:11px;color:#475569;padding:6px 8px;width:60px;text-align:center">Score</th>
        </tr>
      </thead>
      <tbody>{sim_rows}</tbody>
    </table>
  </div>

</div>
"""
    return html


def route_text_complaint(text: str, top_k: int) -> tuple:
    if not text.strip():
        return "<p style='color:red'>Please enter complaint text.</p>", ""
    result = engine.predict(text.strip(), top_k_similar=int(top_k))
    html   = build_output_html(result)
    raw    = json.dumps(result, indent=2, ensure_ascii=False)
    return html, raw


def route_audio_complaint(audio_file, top_k: int) -> tuple:
    if audio_file is None:
        return "<p style='color:red'>Please upload an audio file.</p>", ""
    try:
        result = engine.process(audio_path=audio_file, top_k=int(top_k))
    except ImportError as e:
        return f"<p style='color:orange;font-weight:600'>Dependency Error: {e}</p>", ""
    except Exception as e:
        err_msg = str(e)
        if "ffmpeg" in err_msg.lower() or "winerror 2" in err_msg.lower():
            return (
                "<div style='background:#fffbeb;border:1px solid #fef3c7;padding:16px;border-radius:8px;color:#b45309;line-height:1.5'>"
                "<strong>System Configuration Error: FFmpeg not detected!</strong><br/>"
                "Whisper requires FFmpeg to process and decode audio uploads.<br/><br/>"
                "<strong>To fix this on Windows:</strong><br/>"
                "1. Open PowerShell as Administrator and run: <code>winget install Gyan.FFmpeg</code><br/>"
                "2. <strong>Crucial:</strong> Close and restart your IDE (VS Code), terminal, or command prompt so Windows reloads the new PATH system variable.<br/>"
                "3. Restart the web app and try again."
                "</div>",
                f"Error details: {err_msg}"
            )
        return f"<div style='color:red;padding:12px;border:1px solid #fecaca;background:#fef2f2;border-radius:8px;line-height:1.5'><strong>Error processing audio:</strong> {err_msg}</div>", f"Error details: {err_msg}"
    html = build_output_html(result)
    raw  = json.dumps(result, indent=2, ensure_ascii=False)
    return html, raw


def route_video_complaint(video_file, top_k: int) -> tuple:
    if video_file is None:
        return "<p style='color:red'>Please upload a video file.</p>", ""
    try:
        result = engine.process(video_path=video_file, top_k=int(top_k))
    except ImportError as e:
        return f"<p style='color:orange;font-weight:600'>Dependency Error: {e}</p>", ""
    except Exception as e:
        err_msg = str(e)
        if "ffmpeg" in err_msg.lower() or "winerror 2" in err_msg.lower():
            return (
                "<div style='background:#fffbeb;border:1px solid #fef3c7;padding:16px;border-radius:8px;color:#b45309;line-height:1.5'>"
                "<strong>System Configuration Error: FFmpeg not detected!</strong><br/>"
                "Whisper requires FFmpeg to extract audio from video uploads.<br/><br/>"
                "<strong>To fix this on Windows:</strong><br/>"
                "1. Open PowerShell as Administrator and run: <code>winget install Gyan.FFmpeg</code><br/>"
                "2. <strong>Crucial:</strong> Close and restart your IDE (VS Code), terminal, or command prompt so Windows reloads the new PATH system variable.<br/>"
                "3. Restart the web app and try again."
                "</div>",
                f"Error details: {err_msg}"
            )
        return f"<div style='color:red;padding:12px;border:1px solid #fecaca;background:#fef2f2;border-radius:8px;line-height:1.5'><strong>Error processing video:</strong> {err_msg}</div>", f"Error details: {err_msg}"
    html = build_output_html(result)
    raw  = json.dumps(result, indent=2, ensure_ascii=False)
    return html, raw


def create_app():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("pip install gradio  # then retry")

    EXAMPLE_COMPLAINTS = [
        ["There is a massive pothole on Brigade Road near the hospital causing accidents. URGENT! People are in immediate danger.", 5],
        ["My ration card application has been pending for 45 days. Not urgent, but the matter needs attention when convenient.", 5],
        ["Sewage water overflowing near Central Park. This is a serious problem affecting daily life. Requesting action at the earliest.", 5],
        ["This is a very serious problem. A live electric wire has fallen near the school. People are in immediate danger. Urgent action needed!", 5],
        ["Bus route 42 from East Colony has been suspended for 10 days without notice. Multiple families are affected.", 5],
        ["Illegal construction is happening on government land near the Railway Station. Not urgent but needs attention.", 5],
    ]

    with gr.Blocks(
        title="Complaint Auto-Routing System",
        theme=gr.themes.Soft(),
        css="""
        * { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }
        .gradio-container { max-width: 900px !important; margin: 0 auto; }
        footer { display: none; }
        """,
    ) as demo:

        gr.Markdown("""
# Complaint Auto-Routing System

AI/ML system that automatically routes complaints to the right officer, predicts priority and resolution time, and retrieves similar past complaints — **fully offline, no external APIs**.
        """)

        with gr.Tabs():

            # ── Text Tab
            with gr.Tab("Text Complaint"):
                with gr.Row():
                    text_input = gr.Textbox(
                        label="Complaint Text",
                        placeholder="Describe your complaint here in English…",
                        lines=4,
                    )
                    top_k_text = gr.Slider(1, 10, value=5, step=1, label="Similar complaints")
                text_btn    = gr.Button("Route Complaint", variant="primary")
                text_output = gr.HTML(label="Routing Result")

                with gr.Accordion("Show raw JSON", open=False):
                    text_json = gr.Code(label="JSON", language="json")

                gr.Examples(
                    examples=EXAMPLE_COMPLAINTS,
                    inputs=[text_input, top_k_text],
                )

                text_btn.click(
                    fn=route_text_complaint,
                    inputs=[text_input, top_k_text],
                    outputs=[text_output, text_json],
                )

            # ── Audio Tab
            with gr.Tab("Audio Complaint"):
                gr.Markdown("""
Upload an audio recording of the complaint in English.
**Requires:** `pip install openai-whisper` (local model, English)
                """)
                audio_input = gr.Audio(type="filepath", label="Audio File (.wav, .mp3, .m4a)")
                top_k_audio = gr.Slider(1, 10, value=5, step=1, label="Similar complaints")
                audio_btn   = gr.Button("Transcribe & Route", variant="primary")
                audio_out   = gr.HTML(label="Result")
                audio_json  = gr.Code(label="JSON", language="json")
                audio_btn.click(
                    fn=route_audio_complaint,
                    inputs=[audio_input, top_k_audio],
                    outputs=[audio_out, audio_json],
                )

            # ── Video Tab
            with gr.Tab("Video Complaint"):
                gr.Markdown("""
Upload a video of the complainant speaking.
**Requires:** `pip install openai-whisper` + `ffmpeg` installed system-wide.
                """)
                video_input = gr.Video(label="Video File (.mp4, .mkv, .avi)")
                top_k_video = gr.Slider(1, 10, value=5, step=1, label="Similar complaints")
                video_btn   = gr.Button("Extract Audio & Route", variant="primary")
                video_out   = gr.HTML(label="Result")
                video_json  = gr.Code(label="JSON", language="json")
                video_btn.click(
                    fn=route_video_complaint,
                    inputs=[video_input, top_k_video],
                    outputs=[video_out, video_json],
                )

            # ── About Tab
            with gr.Tab("About"):
                gr.Markdown("""
## Architecture

| Component | Model | Notes |
|-----------|-------|-------|
| **Embeddings** | TF-IDF + SVD (256-dim) | Offline baseline; swap for `paraphrase-multilingual-MiniLM-L12-v2` |
| **Officer Routing** | SVM (RBF kernel) | 8-class, probability calibrated |
| **Priority** | Random Forest | High / Medium / Low |
| **ETA Prediction** | Gradient Boosting Regressor | MAE ≈ 5–8 days |
| **Similarity Search** | Cosine over NumPy matrix | FAISS drop-in available |
| **Audio/Video** | Whisper (local) | English, fully offline |

## Officers
| ID | Name | Department |
|----|------|-----------|
| OFF001 | Rahul Sharma | Infrastructure & Roads |
| OFF002 | Priya Mehta | Water & Sanitation |
| OFF003 | Amit Verma | Electricity & Utilities |
| OFF004 | Sunita Patel | Public Safety & Security |
| OFF005 | Vijay Kumar | Health & Environment |
| OFF006 | Anjali Singh | Land & Property |
| OFF007 | Ravi Nair | Transport & Traffic |
| OFF008 | Meena Reddy | Administrative Services |

## No External APIs
All inference happens locally. Models are trained from scratch on synthetic data.
For production, replace synthetic data with real complaint records.
                """)

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
