"""First-run configuration wizard for the local installation."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from src.runtime.config import AppConfig, DiscordConfig, StorageConfig, save_config
from src.runtime.secrets import SecretStore


class SetupWizard(tk.Toplevel):
    def __init__(self, parent, config: AppConfig, secrets: SecretStore, on_complete):
        super().__init__(parent)
        self.title("Angela setup")
        self.resizable(False, False)
        self.config_model = config
        self.secrets = secrets
        self.on_complete = on_complete

        self.discord_token = tk.StringVar()
        self.google_api_key = tk.StringVar()
        self.channel_id = tk.StringVar(value=config.discord.channel_id or "")
        self.storage_backend = tk.StringVar(value=config.storage.backend)
        self.status = tk.StringVar(value="Enter the required credentials, then save.")

        body = ttk.Frame(self, padding=18)
        body.grid(row=0, column=0, sticky="nsew")
        ttk.Label(body, text="Angela setup", font=("TkDefaultFont", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )
        ttk.Label(
            body,
            text="Credentials are stored using the operating system credential manager.",
            wraplength=420,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 14))

        self._field(body, 2, "Discord bot token", self.discord_token, secret=True)
        self._field(body, 3, "Google/Gemini API key", self.google_api_key, secret=True)
        self._field(body, 4, "Preferred channel ID (optional)", self.channel_id)

        ttk.Label(body, text="Storage").grid(row=5, column=0, sticky="w", pady=6)
        storage_box = ttk.Combobox(
            body,
            textvariable=self.storage_backend,
            values=("sqlite", "postgres"),
            state="readonly",
            width=28,
        )
        storage_box.grid(row=5, column=1, sticky="ew", pady=6)

        ttk.Label(body, textvariable=self.status, wraplength=420).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(12, 8)
        )
        actions = ttk.Frame(body)
        actions.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(actions, text="Save and close", command=self._save).pack(side="right")

        self.transient(parent)
        self.grab_set()

    @staticmethod
    def _field(parent, row, label, variable, secret=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6)
        ttk.Entry(parent, textvariable=variable, show="•" if secret else "", width=42).grid(
            row=row, column=1, sticky="ew", pady=6
        )

    def _save(self):
        try:
            if self.discord_token.get().strip():
                self.secrets.set("discord_token", self.discord_token.get().strip())
            if self.google_api_key.get().strip():
                self.secrets.set("google_api_key", self.google_api_key.get().strip())

            updated = self.config_model.model_copy(
                update={
                    "discord": DiscordConfig(
                        **self.config_model.discord.model_dump(),
                        channel_id=self.channel_id.get().strip() or None,
                    ),
                    "storage": StorageConfig(
                        **self.config_model.storage.model_dump(),
                        backend=self.storage_backend.get(),
                    ),
                }
            )
            save_config(updated)
            self.config_model = updated
            self.status.set("Saved. You can now start Angela.")
            self.on_complete(updated)
            self.after(250, self.destroy)
        except Exception as exc:
            self.status.set(f"Could not save setup: {exc}")
            messagebox.showerror("Angela setup", str(exc), parent=self)

