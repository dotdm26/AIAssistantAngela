import base64
import json
import threading
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-not-found]
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.errors import HttpError  # type: ignore[import-not-found]
from langchain_core.tools import tool

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.labels",
]

BASE_DIR = Path(__file__).resolve().parent
CLIENT_SECRETS_PATH = BASE_DIR / "client_secrets.json"
TOKEN_PATH = BASE_DIR.parent / "tokens" / "token_gmail.json"

_CACHE_LOCK = threading.RLock()
_CACHED_CREDENTIALS = None
_CACHED_GMAIL_SERVICE = None


def _clear_gmail_cache() -> None:
    global _CACHED_CREDENTIALS, _CACHED_GMAIL_SERVICE

    with _CACHE_LOCK:
        _CACHED_CREDENTIALS = None
        _CACHED_GMAIL_SERVICE = None


def _get_credentials() -> Credentials:
    global _CACHED_CREDENTIALS

    with _CACHE_LOCK:
        if _CACHED_CREDENTIALS and _CACHED_CREDENTIALS.valid:
            return _CACHED_CREDENTIALS

    creds = None

    if not CLIENT_SECRETS_PATH.exists():
        raise FileNotFoundError(
            "client_secrets.json was not found next to this script. "
            f"Expected at: {CLIENT_SECRETS_PATH}"
        )

    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
            try:
                # Works better in remote/WSL sessions where browser auto-launch may fail.
                creds = flow.run_local_server(port=0, open_browser=False)
            except Exception:
                creds = flow.run_console()

        with open(TOKEN_PATH, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    with _CACHE_LOCK:
        _CACHED_CREDENTIALS = creds

    return creds


def _get_gmail_service():
    global _CACHED_GMAIL_SERVICE

    creds = _get_credentials()

    with _CACHE_LOCK:
        if _CACHED_GMAIL_SERVICE is not None and _CACHED_CREDENTIALS is creds:
            return _CACHED_GMAIL_SERVICE

        service = build("gmail", "v1", credentials=creds)
        _CACHED_GMAIL_SERVICE = service
        return service


def _is_auth_failure(error: Exception) -> bool:
    if isinstance(error, HttpError):
        status_code = getattr(error.resp, "status", None)
        if status_code in {401, 403}:
            return True

    error_text = str(error).lower()
    return "invalid_grant" in error_text or "unauthorized" in error_text


def _execute_gmail_request(operation_name: str, request_factory):
    del operation_name

    for attempt in range(2):
        try:
            request = request_factory()
            return request.execute()
        except Exception as error:
            if attempt == 0 and _is_auth_failure(error):
                _clear_gmail_cache()
                continue
            raise


def _parse_max_results(max_results: int, lower_bound: int, upper_bound: int) -> int:
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        value = lower_bound
    return max(lower_bound, min(upper_bound, value))


def _message_title(message: dict) -> str:
    payload = message.get("payload", {}) or {}
    headers = payload.get("headers", []) or []
    for header in headers:
        if (header.get("name") or "").lower() == "subject":
            return header.get("value") or "(no subject)"
    return message.get("snippet") or "(no subject)"


def _message_to_dict(message: dict) -> dict:
    payload = message.get("payload", {}) or {}
    headers = payload.get("headers", []) or []
    header_map = {str(header.get("name", "")).lower(): header.get("value", "") for header in headers}
    return {
        "id": message.get("id"),
        "threadId": message.get("threadId", ""),
        "title": _message_title(message),
        "from": header_map.get("from", ""),
        "to": header_map.get("to", ""),
        "date": header_map.get("date", ""),
        "snippet": message.get("snippet", ""),
        "labelIds": message.get("labelIds", []),
    }


def _list_messages(user_id: str, query: Optional[str], max_results: int, label_ids: Optional[list] = None):
    safe_max = _parse_max_results(max_results, 1, 50)
    service = _get_gmail_service()

    list_kwargs = {
        "userId": user_id,
        "maxResults": safe_max,
    }
    if query:
        list_kwargs["q"] = query
    if label_ids:
        list_kwargs["labelIds"] = label_ids

    response = _execute_gmail_request(
        "list_messages",
        lambda: service.users().messages().list(**list_kwargs),
    )
    messages = response.get("messages", [])

    items = []
    for item in messages:
        message = _execute_gmail_request(
            "get_message",
            lambda item_id=item["id"]: service.users().messages().get(
                userId=user_id,
                id=item_id,
                format="metadata",
                metadataHeaders=["Subject", "From", "To", "Date"],
            ),
        )
        items.append(_message_to_dict(message))

    return {"count": len(items), "messages": items}


def _label_to_dict(label: dict) -> dict:
    return {
        "id": label.get("id", ""),
        "name": label.get("name", ""),
        "type": label.get("type", "user"),
        "messageListVisibility": label.get("messageListVisibility", ""),
        "labelListVisibility": label.get("labelListVisibility", ""),
    }


@tool
def gmail_list_messages(user_id: str = "me", query: Optional[str] = None, max_results: int = 20) -> str:
    """List Gmail messages with message titles, snippets, and metadata."""
    try:
        payload = _list_messages(user_id=user_id, query=query, max_results=max_results, label_ids=["INBOX"])
        return json.dumps(payload)
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_search_messages(query: str, user_id: str = "me", max_results: int = 20) -> str:
    """Search Gmail messages using Gmail query syntax and return message titles."""
    try:
        if not (query or "").strip():
            return json.dumps({"error": "query is required."})

        payload = _list_messages(user_id=user_id, query=query, max_results=max_results)
        return json.dumps(payload)
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_list_labels(user_id: str = "me") -> str:
    """List all Gmail labels available in the mailbox."""
    try:
        response = _execute_gmail_request(
            "list_labels",
            lambda: _get_gmail_service().users().labels().list(userId=user_id),
        )
        labels = [_label_to_dict(label) for label in response.get("labels", [])]
        return json.dumps({"count": len(labels), "labels": labels})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_create_label(
    name: str,
    user_id: str = "me",
    message_list_visibility: str = "show",
    label_list_visibility: str = "labelShow",
) -> str:
    """Create a new Gmail label."""
    try:
        if not (name or "").strip():
            return json.dumps({"error": "name is required."})

        label_body = {
            "name": name.strip(),
            "messageListVisibility": message_list_visibility,
            "labelListVisibility": label_list_visibility,
        }
        label = _execute_gmail_request(
            "create_label",
            lambda: _get_gmail_service().users().labels().create(userId=user_id, body=label_body),
        )
        return json.dumps({"label": _label_to_dict(label)})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_update_label(
    label_id: str,
    user_id: str = "me",
    name: Optional[str] = None,
    message_list_visibility: Optional[str] = None,
    label_list_visibility: Optional[str] = None,
) -> str:
    """Update an existing Gmail label."""
    try:
        if not (label_id or "").strip():
            return json.dumps({"error": "label_id is required."})

        existing = _execute_gmail_request(
            "get_label",
            lambda: _get_gmail_service().users().labels().get(userId=user_id, id=label_id),
        )
        label_body = dict(existing)

        if name is not None:
            label_body["name"] = name.strip()
        if message_list_visibility is not None:
            label_body["messageListVisibility"] = message_list_visibility
        if label_list_visibility is not None:
            label_body["labelListVisibility"] = label_list_visibility

        label = _execute_gmail_request(
            "update_label",
            lambda: _get_gmail_service().users().labels().update(userId=user_id, id=label_id, body=label_body),
        )
        return json.dumps({"label": _label_to_dict(label)})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_delete_label(label_id: str, user_id: str = "me") -> str:
    """Delete a Gmail label."""
    try:
        if not (label_id or "").strip():
            return json.dumps({"error": "label_id is required."})

        _execute_gmail_request(
            "delete_label",
            lambda: _get_gmail_service().users().labels().delete(userId=user_id, id=label_id),
        )
        return json.dumps({"status": "success", "message": f"Label {label_id} deleted."})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_add_label_to_message(message_id: str, label_id: str, user_id: str = "me") -> str:
    """Add an existing label to a Gmail message."""
    try:
        if not (message_id or "").strip() or not (label_id or "").strip():
            return json.dumps({"error": "message_id and label_id are required."})

        body = {"addLabelIds": [label_id], "removeLabelIds": []}
        message = _execute_gmail_request(
            "modify_message_add_label",
            lambda: _get_gmail_service().users().messages().modify(userId=user_id, id=message_id, body=body),
        )
        return json.dumps({"id": message.get("id"), "labelIds": message.get("labelIds", [])})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_remove_label_from_message(message_id: str, label_id: str, user_id: str = "me") -> str:
    """Remove an existing label from a Gmail message."""
    try:
        if not (message_id or "").strip() or not (label_id or "").strip():
            return json.dumps({"error": "message_id and label_id are required."})

        body = {"addLabelIds": [], "removeLabelIds": [label_id]}
        message = _execute_gmail_request(
            "modify_message_remove_label",
            lambda: _get_gmail_service().users().messages().modify(userId=user_id, id=message_id, body=body),
        )
        return json.dumps({"id": message.get("id"), "labelIds": message.get("labelIds", [])})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_create_draft(sender: Optional[str] = None, recipient: Optional[str] = None, subject: Optional[str] = None, message_text: Optional[str] = None) -> str:
    """Create and insert a draft email."""
    try:
        service = _get_gmail_service()

        message = EmailMessage()
        message.set_content(message_text or "")
        if sender:
            message["From"] = sender
        if recipient:
            message["To"] = recipient
        if subject:
            message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"message": {"raw": encoded_message}}
        draft = _execute_gmail_request(
            "create_draft",
            lambda: service.users().drafts().create(userId="me", body=create_message),
        )
        return json.dumps({"id": draft.get("id"), "messageId": draft.get("message", {}).get("id", "")})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})


@tool
def gmail_send_message(sender: Optional[str] = None, recipient: Optional[str] = None, subject: Optional[str] = None, message_text: Optional[str] = None) -> str:
    """Create and send an email message."""
    try:
        service = _get_gmail_service()

        message = EmailMessage()
        message.set_content(message_text or "")
        if sender:
            message["From"] = sender
        if recipient:
            message["To"] = recipient
        if subject:
            message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        create_message = {"raw": encoded_message}
        send_message = _execute_gmail_request(
            "send_message",
            lambda: service.users().messages().send(userId="me", body=create_message),
        )
        return json.dumps({"id": send_message.get("id"), "threadId": send_message.get("threadId", "")})
    except HttpError as error:
        return json.dumps({"error": f"Google Gmail API error: {error}"})
    except Exception as error:
        return json.dumps({"error": str(error)})

gmail_label_tools_list = [
    gmail_list_labels,
    gmail_create_label,
    gmail_update_label,
    gmail_delete_label,
    gmail_add_label_to_message,
    gmail_remove_label_from_message,
]

gmail_read_tools_list = [
    gmail_list_messages,
    gmail_search_messages
] 

gmail_write_tools_list = [
    gmail_create_draft,
    gmail_send_message,
]

# Backward-compatible aggregate list used by existing global tool wiring.
gmail_tools_list = gmail_read_tools_list + gmail_label_tools_list + gmail_write_tools_list