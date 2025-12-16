Tech stack
- **Languages:** Python, JavaScript (Node.js)
- **Web:** Flask, REST, HTML/CSS
- **Data & ML:** (add your stack: e.g., PyTorch/TensorFlow, SQL, Redis)
- **Tools:** Docker, Git, CI (GitHub Actions), testing (pytest)

Highlights & strengths
- **System design:** Experience designing services with clear SLAs, caching, and horizontal scaling.
- **Production code:** Emphasis on tests, CI, monitoring, and observability.
- **Product sense:** Ship features with strong UX and measurable impact.
- **Collaboration:** Code reviews, mentoring, and cross-team communication.

This repository — Legal Chat App
- **What it is:** A minimal legal assistant web app demonstrating end-to-end design: prompt templates, a backend service, and a simple chat UI.
- **Key files:**
  - `legal_app.py` — app entrypoint (Flask server)
  - `prompt_template.py` — templated prompts used by the assistant
  - `templates/chat.html` — front-end demo UI
- **Why it matters:** Shows architecture decisions (separation of concerns), prompt-driven logic, and a deployable web demo.

Quick run (local)
1. Create virtualenv and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app locally:

```bash
python legal_app.py
```

3. Open the demo in your browser at `http://127.0.0.1:5000/` (or as printed by the server).