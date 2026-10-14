"""Secret storage with OS keyring support and environment fallback."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class SecretStore(ABC):
    @abstractmethod
    def get(self, name: str) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, name: str, value: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> None:
        raise NotImplementedError


_ENV_NAMES = {
    "discord_token": "DISCORD_TOKEN",
    "google_api_key": "GOOGLE_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
}


class KeyringSecretStore(SecretStore):
    def __init__(self, service_name: str = "Angela"):
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - exercised by minimal installs
            raise RuntimeError("The keyring package is required for desktop secret storage.") from exc
        self._keyring = keyring
        self.service_name = service_name

    def get(self, name: str) -> str | None:
        try:
            value = self._keyring.get_password(self.service_name, name)
        except self._keyring.errors.KeyringError:
            value = None
        return value or os.getenv(_ENV_NAMES.get(name, name.upper())) or None

    def set(self, name: str, value: str) -> None:
        if not value:
            self.delete(name)
            return
        try:
            self._keyring.set_password(self.service_name, name, value)
        except self._keyring.errors.KeyringError as exc:
            raise RuntimeError("No usable operating-system credential manager is available.") from exc

    def delete(self, name: str) -> None:
        try:
            self._keyring.delete_password(self.service_name, name)
        except self._keyring.errors.KeyringError:
            pass


class EnvironmentSecretStore(SecretStore):
    """Development/Docker fallback; never persists secret values."""

    def get(self, name: str) -> str | None:
        return os.getenv(_ENV_NAMES.get(name, name.upper())) or None

    def set(self, name: str, value: str) -> None:
        raise RuntimeError("Environment-backed secrets cannot be written by the application.")

    def delete(self, name: str) -> None:
        raise RuntimeError("Environment-backed secrets cannot be deleted by the application.")


def get_secret_store(*, desktop: bool = False) -> SecretStore:
    if desktop:
        try:
            return KeyringSecretStore()
        except RuntimeError:
            pass
    return EnvironmentSecretStore()

