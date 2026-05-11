"""
Gradio demo for AI Text Trust Service.

Runs as a standalone app that calls the FastAPI backend.
If backend is unavailable — uses detectors directly (offline mode).

Usage:
  # С поднятым API (make docker-up или make run), затем:
  make demo
  # Или: AITRUST_DEMO_API_BASE=http://host:8000 uv run python demo/app.py

  # Без бэкенда — только TF-IDF + explainer в процессе Gradio:
  make demo-offline
"""

import argparse
import os
import sys
from pathlib import Path

import gradio as gr

# Базовый URL FastAPI (без завершающего слэша). Переопределение: AITRUST_DEMO_API_BASE
DEFAULT_API_BASE = os.environ.get("AITRUST_DEMO_API_BASE", "http://localhost:8000").rstrip("/")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# ── Detector backends ─────────────────────────────────────────

_detector = None
_explainer = None
_preprocessor = None


def _init_offline():
    """Load models directly (no FastAPI backend)."""
    global _detector, _explainer, _preprocessor

    from app.config import Settings
    from app.detectors.tfidf import TFIDFDetector
    from app.explanation.markers import StyleExplainer
    from app.preprocessing.pipeline import TextPreprocessor

    settings = Settings()
    _preprocessor = TextPreprocessor(settings)

    _detector = TFIDFDetector(settings.artifacts_dir)
    _detector.load()

    baselines_path = settings.artifacts_dir / "human_baselines.json"
    _explainer = StyleExplainer(baselines_path)
    _explainer.load()


def _analyze_offline(text: str) -> dict:
    """Run detection directly without HTTP."""
    prep = _preprocessor(text)

    if prep.cleaned_length < 300:
        return {
            "status": "NO_DECISION",
            "reason": f"Текст слишком короткий ({prep.cleaned_length} символов, минимум 300)",
        }

    result = _detector.predict(prep.text)
    confidence = result.prob_ai
    verdict = "ai" if confidence >= 0.5 else "human"

    if confidence >= 0.70:
        risk = "HIGH"
    elif confidence >= 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    explanation = None
    if _explainer and _explainer.is_ready():
        explanation = _explainer.explain(prep.text)

    return {
        "status": "SUCCESS",
        "verdict": verdict,
        "confidence": confidence,
        "risk_level": risk,
        "inference_ms": result.inference_ms,
        "text_length": prep.original_length,
        "was_truncated": prep.was_truncated,
        "explanation": explanation,
    }


def _analyze_online(text: str) -> dict:
    """Call FastAPI backend via HTTP."""
    import httpx

    try:
        r = httpx.post(
            f"{DEFAULT_API_BASE}/api/v1/analyze",
            json={"text": text, "explain": True},
            timeout=10.0,
        )
        return r.json()
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


# ── UI logic ──────────────────────────────────────────────────

RISK_COLORS = {
    "HIGH": "🔴",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}

RISK_LABELS = {
    "HIGH": "Высокая вероятность AI",
    "MEDIUM": "Зона неопределённости",
    "LOW": "Вероятно, человек",
}


def analyze(text: str, mode: str) -> tuple[str, str, str]:
    """Main analysis function. Returns (verdict_html, details, explanation)."""
    if not text or not text.strip():
        return "❌ Введите текст", "", ""

    if mode == "API (backend)":
        data = _analyze_online(text)
    else:
        data = _analyze_offline(text)

    if data.get("status") == "NO_DECISION":
        reason = data.get("reason", "Текст слишком короткий")
        return f"⚠️ {reason}", "", ""

    if data.get("status") == "ERROR":
        return f"❌ Ошибка: {data.get('reason', 'unknown')}", "", ""

    verdict = data["verdict"]
    confidence = data["confidence"]
    risk = data.get("risk_level", "MEDIUM")
    icon = RISK_COLORS.get(risk, "⚪")
    risk_label = RISK_LABELS.get(risk, risk)

    # Verdict block
    verdict_text = "Искусственный интеллект" if verdict == "ai" else "Человек"
    verdict_html = f"""
    <div style="text-align: center; padding: 20px;">
        <h2>{icon} {verdict_text}</h2>
        <p style="font-size: 18px;">Уверенность: <b>{confidence:.1%}</b></p>
        <p style="color: gray;">{risk_label}</p>
    </div>
    """

    # Details
    details_parts = [f"Длина текста: {data.get('text_length', len(text))} символов"]
    if data.get("was_truncated"):
        details_parts.append("⚠️ Текст был обрезан до 8000 символов")
    if "inference_ms" in data:
        details_parts.append(f"Время inference: {data['inference_ms']:.1f} ms")
    elif "processing_time_ms" in data:
        details_parts.append(f"Время обработки: {data['processing_time_ms']:.1f} ms")
    details = "\n".join(details_parts)

    # Explanation
    explanation_text = ""
    expl = data.get("explanation")
    if expl:
        if isinstance(expl, dict):
            summary = expl.get("summary", "")
            markers = expl.get("top_markers", [])
        else:
            summary = expl.summary
            markers = expl.top_markers

        if summary:
            explanation_text = f"📊 {summary}\n\n"

        if markers:
            explanation_text += "Детали отклонений:\n"
            for m in markers:
                if isinstance(m, dict):
                    feat = m["feature"]
                    val = m["value"]
                    base = m["human_baseline"]
                    dev = m["deviation_percent"]
                    direction = m["direction"]
                else:
                    feat, val, base, dev, direction = (
                        m.feature,
                        m.value,
                        m.human_baseline,
                        m.deviation_percent,
                        m.direction,
                    )
                arrow = "↑" if direction == "above" else "↓"
                explanation_text += (
                    f"  {arrow} {feat}: {val:.3f} (норма: {base:.3f}, отклонение: {dev:+.0f}%)\n"
                )

    return verdict_html, details, explanation_text


# ── Gradio app ────────────────────────────────────────────────


def build_app(offline: bool = False) -> gr.Blocks:
    default_mode = "Offline (прямой)" if offline else "API (backend)"

    with gr.Blocks(title="AI Text Trust Service") as app:
        gr.Markdown(
            """
            # 🔍 AI Text Trust Service
            **Детекция текстов, сгенерированных ИИ, на русском языке**

            Вставьте текст (минимум 300 символов) и нажмите «Проверить».
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Текст для анализа",
                    placeholder="Вставьте текст на русском языке...",
                    lines=10,
                    max_lines=20,
                )
                with gr.Row():
                    mode = gr.Radio(
                        choices=["API (backend)", "Offline (прямой)"],
                        value=default_mode,
                        label="Режим",
                    )
                    btn = gr.Button("🔍 Проверить", variant="primary", scale=2)

            with gr.Column(scale=1):
                verdict_out = gr.HTML(label="Вердикт")
                details_out = gr.Textbox(label="Детали", lines=3, interactive=False)
                explanation_out = gr.Textbox(
                    label="Объяснение (стилометрия)", lines=8, interactive=False
                )

        btn.click(
            fn=analyze,
            inputs=[text_input, mode],
            outputs=[verdict_out, details_out, explanation_out],
        )

        gr.Markdown(
            """
            ---
            *Результат носит вероятностный характер и не является доказательством.*

            **Метод:** TF-IDF (char 3-5 + word 1-2) + LogReg, 50K признаков, обучено на 10K текстов.
            Cascade: TF-IDF → Transformer (ruRoBERTa-large) при неуверенности.
            """
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio demo")
    parser.add_argument("--offline", action="store_true", help="Run without FastAPI backend")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    if args.offline:
        print("Loading models for offline mode...")
        _init_offline()
        print("Models loaded.")

    app = build_app(offline=args.offline)
    app.launch(server_port=args.port, share=args.share)
