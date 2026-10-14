"""Small desktop controller for the Angela worker.

Tkinter keeps the first controller dependency-light. The worker remains a
separate process, which makes this controller replaceable by a future Tauri
shell without changing the assistant runtime.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from src.runtime.config import load_config
from src.runtime.diagnostics import run_diagnostics
from src.runtime.secrets import get_secret_store
from src.desktop.setup_wizard import SetupWizard


class AngelaDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Angela")
        self.root.geometry("560x430")
        self.root.minsize(520, 380)
        self.config = load_config()
        self.secrets = get_secret_store(desktop=True)
        self.worker: subprocess.Popen | None = None
        self.events: queue.Queue[dict] = queue.Queue()
        self.tray_icon = None
        self.status = tk.StringVar(value="Stopped")
        self.details = tk.StringVar(value="Angela is not running.")
        self.health_text = tk.Text(root, height=14, state="disabled", wrap="word")
        self._build_ui()
        self._setup_tray()
        self.root.after(150, self._poll_events)
        self.root.protocol("WM_DELETE_WINDOW", self._hide_window)
        self.root.after(250, self._maybe_open_setup)

    def _maybe_open_setup(self):
        if not self.secrets.get("discord_token") or not self.secrets.get("google_api_key"):
            self.open_setup()

    def _setup_tray(self):
        """Create a tray icon when the platform supports pystray."""
        try:
            import pystray
            from PIL import Image, ImageDraw
        except ImportError:
            return

        image = Image.new("RGBA", (64, 64), (36, 99, 235, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((12, 12, 52, 52), fill=(255, 255, 255, 255))
        draw.ellipse((22, 25, 28, 31), fill=(36, 99, 235, 255))
        draw.ellipse((36, 25, 42, 31), fill=(36, 99, 235, 255))
        self.tray_icon = pystray.Icon(
            "angela",
            image,
            "Angela",
            menu=pystray.Menu(
                pystray.MenuItem("Open Angela", lambda: self.root.after(0, self._show_window)),
                pystray.MenuItem("Start", lambda: self.root.after(0, self.start_worker)),
                pystray.MenuItem("Stop", lambda: self.root.after(0, self.stop_worker)),
                pystray.MenuItem("Quit", lambda: self.root.after(0, self._quit)),
            ),
        )
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def _show_window(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _build_ui(self):
        body = ttk.Frame(self.root, padding=18)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Angela", font=("TkDefaultFont", 20, "bold")).pack(anchor="w")
        ttk.Label(body, textvariable=self.status, font=("TkDefaultFont", 12)).pack(anchor="w", pady=(4, 0))
        ttk.Label(body, textvariable=self.details, wraplength=500).pack(anchor="w", pady=(4, 14))

        actions = ttk.Frame(body)
        actions.pack(anchor="w", pady=(0, 12))
        ttk.Button(actions, text="Start Angela", command=self.start_worker).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Stop", command=self.stop_worker).pack(side="left", padx=6)
        ttk.Button(actions, text="Setup", command=self.open_setup).pack(side="left", padx=6)
        ttk.Button(actions, text="Diagnostics", command=self.show_diagnostics).pack(side="left", padx=6)

        ttk.Label(body, text="Worker status").pack(anchor="w")
        self.health_text.pack(fill="both", expand=True)

    def open_setup(self):
        SetupWizard(self.root, self.config, self.secrets, self._config_updated)

    def _config_updated(self, config):
        self.config = config
        self.details.set("Configuration saved. Start Angela when ready.")

    def start_worker(self):
        if self.worker and self.worker.poll() is None:
            return
        root_dir = Path(__file__).resolve().parents[2]
        worker_environment = os.environ.copy()
        worker_environment["ANGELA_DESKTOP"] = "1"
        self.worker = subprocess.Popen(
            [sys.executable, "-m", "src.runtime.worker"],
            cwd=root_dir,
            env=worker_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.status.set("Starting")
        self.details.set("Angela is initializing its local database and connections.")
        threading.Thread(target=self._read_worker_output, daemon=True).start()

    def _read_worker_output(self):
        if not self.worker or not self.worker.stdout:
            return
        for line in self.worker.stdout:
            try:
                self.events.put(json.loads(line))
            except json.JSONDecodeError:
                self.events.put({"type": "log", "message": line.rstrip()})

    def stop_worker(self):
        if not self.worker or self.worker.poll() is not None:
            self.status.set("Stopped")
            return
        self.status.set("Stopping")
        self.worker.terminate()

    def _poll_events(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)
        if self.worker and self.worker.poll() is not None and self.status.get() not in {"Stopped", "Failed"}:
            self.status.set("Stopped" if self.worker.returncode == 0 else "Failed")
        self.root.after(150, self._poll_events)

    def _handle_event(self, event):
        event_type = event.get("type")
        if event_type == "state":
            state = event.get("state", "unknown").title()
            self.status.set(state)
            if event.get("message"):
                self.details.set(event["message"])
        elif event_type == "health":
            self._append_health(event)
            if event.get("name") == "runtime" and event.get("state") == "failed":
                self.status.set("Failed")
                self.details.set(event.get("message") or "Runtime failed")
        elif event_type == "log":
            self._append_health(event)

    def _append_health(self, event):
        self.health_text.configure(state="normal")
        self.health_text.insert("end", json.dumps(event, ensure_ascii=False) + "\n")
        self.health_text.see("end")
        self.health_text.configure(state="disabled")

    def show_diagnostics(self):
        results = run_diagnostics(
            self.config,
            discord_configured=bool(self.secrets.get("discord_token")),
            llm_configured=bool(self.secrets.get("google_api_key")),
        )
        messagebox.showinfo(
            "Angela diagnostics",
            "\n".join(f"{item['state'].upper()}: {item['name']} — {item['message']}" for item in results),
            parent=self.root,
        )

    def _hide_window(self):
        if self.tray_icon is not None:
            self.root.withdraw()
            return
        self._quit()

    def _quit(self):
        self.stop_worker()
        if self.tray_icon is not None:
            self.tray_icon.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = AngelaDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

