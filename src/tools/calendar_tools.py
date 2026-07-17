import datetime
import os.path
from pathlib import Path
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_core.tools import tool

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


BASE_DIR = Path(__file__).resolve().parent
CLIENT_SECRETS_PATH = BASE_DIR / "client_secrets.json"
TOKEN_PATH = BASE_DIR / "token.json"

def _get_credentials() -> Credentials:
  creds = None

  if not CLIENT_SECRETS_PATH.exists():
    raise FileNotFoundError(
        "client_secrets.json was not found next to this script. "
        f"Expected at: {CLIENT_SECRETS_PATH}"
    )

  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
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
      creds = flow.run_local_server(port=0)

    with open(TOKEN_PATH, "w") as token:
      token.write(creds.to_json())

  return creds


def _get_calendar_service():
  creds = _get_credentials()
  return build("calendar", "v3", credentials=creds)


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


@tool
def list_calendars() -> str:
  """List calendars accessible by the configured Google account."""
  try:
    service = _get_calendar_service()
    response = service.calendarList().list().execute()
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
    service = _get_calendar_service()

    now = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=now,
            maxResults=safe_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
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
    service = _get_calendar_service()
    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=start_iso,
            timeMax=end_iso,
            maxResults=safe_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])
    payload = [_event_to_dict(event) for event in events]
    return json.dumps({"count": len(payload), "events": payload})
  except HttpError as error:
    return json.dumps({"error": f"Google Calendar API error: {error}"})
  except Exception as error:
    return json.dumps({"error": str(error)})
