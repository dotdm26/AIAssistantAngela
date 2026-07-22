import datetime
import threading
from pathlib import Path
import json
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import-not-found]
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from googleapiclient.errors import HttpError  # type: ignore[import-not-found]
from langchain_core.tools import tool

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar"]

BASE_DIR = Path(__file__).resolve().parent
CLIENT_SECRETS_PATH = BASE_DIR / "client_secrets.json"
TOKEN_PATH = BASE_DIR.parent / "tokens" / "token_calendar.json"

_CACHE_LOCK = threading.RLock()
_CACHED_CREDENTIALS = None
_CACHED_CALENDAR_SERVICE = None


def _clear_calendar_cache() -> None:
  global _CACHED_CREDENTIALS, _CACHED_CALENDAR_SERVICE

  with _CACHE_LOCK:
    _CACHED_CREDENTIALS = None
    _CACHED_CALENDAR_SERVICE = None


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

  # Token stores the user's access and refresh credentials for calendar scopes.
  TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)

  if TOKEN_PATH.exists():
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          str(CLIENT_SECRETS_PATH), SCOPES
      )
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


def _get_calendar_service():
  global _CACHED_CALENDAR_SERVICE

  creds = _get_credentials()

  with _CACHE_LOCK:
    if _CACHED_CALENDAR_SERVICE is not None and _CACHED_CREDENTIALS is creds:
      return _CACHED_CALENDAR_SERVICE

    service = build("calendar", "v3", credentials=creds)
    _CACHED_CALENDAR_SERVICE = service
    return service


def _is_auth_failure(error: Exception) -> bool:
  if isinstance(error, HttpError):
    status_code = getattr(error.resp, "status", None)
    if status_code in {401, 403}:
      return True

  error_text = str(error).lower()
  return "invalid_grant" in error_text or "unauthorized" in error_text


def _execute_calendar_request(operation_name: str, request_factory):
  for attempt in range(2):
    try:
      request = request_factory()
      return request.execute()
    except Exception as error:
      if attempt == 0 and _is_auth_failure(error):
        _clear_calendar_cache()
        continue
      raise


def _event_to_dict(event: dict) -> dict:
  start = event.get("start", {}).get("dateTime", event.get("start", {}).get("date"))
  end = event.get("end", {}).get("dateTime", event.get("end", {}).get("date"))
  return {
      "id": event.get("id"),
      "summary": event.get("summary", "(no title)"),
      "start": start,
      "end": end,
      "location": event.get("location", ""),
      "htmlLink": event.get("htmlLink", ""),
  }


def _build_reminders_payload(reminders: Optional[list]) -> Optional[dict]:
  if reminders is None:
    return None

  overrides = []
  for item in reminders:
    if isinstance(item, dict):
      method = str(item.get("method", "popup")).lower()
      minutes = item.get("minutes")
    else:
      method = "popup"
      minutes = item

    if method not in {"popup", "email"}:
      raise ValueError("Reminder method must be 'popup' or 'email'.")

    try:
      minutes_int = int(minutes)
    except (TypeError, ValueError):
      raise ValueError("Reminder minutes must be an integer.")

    if minutes_int < 0 or minutes_int > 40320:
      raise ValueError("Reminder minutes must be between 0 and 40320.")

    overrides.append({"method": method, "minutes": minutes_int})

  return {"useDefault": False, "overrides": overrides}

@tool
def list_calendars() -> str:
  """List calendars accessible by the configured Google account."""
  try:
    response = _execute_calendar_request(
        "list_calendars",
        lambda: _get_calendar_service().calendarList().list(),
    )
    calendars = response.get("items", [])

    payload = [
        {
            "id": cal.get("id"),
            "summary": cal.get("summary", ""),
            "primary": bool(cal.get("primary", False)),
            "timeZone": cal.get("timeZone", ""),
        }
        for cal in calendars
    ]
    return json.dumps({"count": len(payload), "calendars": payload})
  except HttpError as error:
    return json.dumps({"error": f"Google Calendar API error: {error}"})
  except Exception as error:
    return json.dumps({"error": str(error)})


@tool
def get_upcoming_calendar_events(max_results: int = 10, calendar_id: str = "primary") -> str:
  """Get upcoming events from a calendar (default: primary)."""
  try:
    safe_max = max(1, min(50, int(max_results)))

    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    events_result = _execute_calendar_request(
      "get_upcoming_calendar_events",
      lambda: _get_calendar_service()
      .events()
      .list(
        calendarId=calendar_id,
        timeMin=now,
        maxResults=safe_max,
        singleEvents=True,
        orderBy="startTime",
      ),
    )
    events = events_result.get("items", [])

    payload = [_event_to_dict(event) for event in events]
    return json.dumps({"count": len(payload), "events": payload})
  except HttpError as error:
    return json.dumps({"error": f"Google Calendar API error: {error}"})
  except Exception as error:
    return json.dumps({"error": str(error)})


@tool
def get_calendar_events_between(
    start_iso: str,
    end_iso: str,
    calendar_id: str = "primary",
    max_results: int = 50,
) -> str:
  """
  Get events between two ISO datetime values.
  Example start_iso: 2026-07-17T00:00:00+00:00
  """
  try:
    safe_max = max(1, min(250, int(max_results)))
    events_result = _execute_calendar_request(
      "get_calendar_events_between",
      lambda: _get_calendar_service()
      .events()
      .list(
        calendarId=calendar_id,
        timeMin=start_iso,
        timeMax=end_iso,
        maxResults=safe_max,
        singleEvents=True,
        orderBy="startTime",
      ),
    )
    events = events_result.get("items", [])
    payload = [_event_to_dict(event) for event in events]
    return json.dumps({"count": len(payload), "events": payload})
  except HttpError as error:
    return json.dumps({"error": f"Google Calendar API error: {error}"})
  except Exception as error:
    return json.dumps({"error": str(error)})

@tool
def create_calendar_event(calendar_id: str = "primary", summary: str = "", description: Optional[str] = None, 
                          start_iso: str = "", end_iso: str = "", attendees: Optional[list] = None, 
                          location: Optional[str] = None, reminders: Optional[list] = None,
                          title: Optional[str] = None) -> str:
    """
    Create a new event in the specified calendar.
    """
    try:
      resolved_summary = summary if summary else (title or "")
      reminders_payload = _build_reminders_payload(reminders)
      event_body = {
          "summary": resolved_summary,
          "description": description,
          "start": {"dateTime": start_iso},
          "end": {"dateTime": end_iso},
          "attendees": [{"email": email} for email in attendees] if attendees else [],
          "location": location,
        "reminders": reminders_payload if reminders_payload is not None else {"useDefault": True},
      }
      event = _execute_calendar_request(
          "create_calendar_event",
          lambda: _get_calendar_service().events().insert(calendarId=calendar_id, body=event_body),
      )
      return json.dumps({"id": event.get("id"), "htmlLink": event.get("htmlLink")})
    except HttpError as error:
      return json.dumps({"error": f"Google Calendar API error: {error}"})
    except Exception as error:
      return json.dumps({"error": str(error)})

@tool 
def update_calendar_event(calendar_id: str = "primary", event_id: str = "", summary: Optional[str] = None, description: Optional[str] = None, 
                          start_iso: Optional[str] = None, end_iso: Optional[str] = None, attendees: Optional[list] = None, 
                          location: Optional[str] = None, reminders: Optional[list] = None,
                          title: Optional[str] = None) -> str:
    """
    Update an existing event in the specified calendar.
    """
    try:
      if not event_id:
        return json.dumps({"error": "event_id is required to update an event."})

      event_body = {}
      resolved_summary = summary if summary is not None else title
      if resolved_summary is not None:
        event_body["summary"] = resolved_summary
      if description is not None:
        event_body["description"] = description
      if start_iso is not None:
        event_body["start"] = {"dateTime": start_iso}
      if end_iso is not None:
        event_body["end"] = {"dateTime": end_iso}
      if attendees is not None:
        event_body["attendees"] = [{"email": email} for email in attendees]
      if location is not None:
        event_body["location"] = location
      if reminders is not None:
        event_body["reminders"] = _build_reminders_payload(reminders)

      if not event_body:
        return json.dumps({"error": "No fields provided to update."})

      event = _execute_calendar_request(
          "update_calendar_event",
          lambda: _get_calendar_service().events().patch(calendarId=calendar_id, eventId=event_id, body=event_body),
      )
      return json.dumps({"id": event.get("id"), "htmlLink": event.get("htmlLink"), "reminders": event.get("reminders", {})})
    except HttpError as error:
      return json.dumps({"error": f"Google Calendar API error: {error}"})
    except Exception as error:
      return json.dumps({"error": str(error)})

@tool
def delete_calendar_event(calendar_id: str = "primary", event_id: str = "") -> str:
    """
    Delete an existing event in the specified calendar.
    """
    try:
      _execute_calendar_request(
          "delete_calendar_event",
          lambda: _get_calendar_service().events().delete(calendarId=calendar_id, eventId=event_id),
      )
      return json.dumps({"status": "success", "message": f"Event {event_id} deleted."})
    except HttpError as error:
      return json.dumps({"error": f"Google Calendar API error: {error}"})
    except Exception as error:
      return json.dumps({"error": str(error)})


calendar_tools_list = [
  list_calendars,
  get_upcoming_calendar_events,
  get_calendar_events_between,
  create_calendar_event,
  update_calendar_event,
  delete_calendar_event,
]