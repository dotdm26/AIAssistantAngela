from datetime import datetime
import json
import zoneinfo
from langchain.tools import tool

@tool
def get_time() -> str:
  """Get the current date and human-readable time for Europe/London timezone."""
  try:
    now = datetime.now(tz=zoneinfo.ZoneInfo("Europe/London"))
    date_text = now.strftime("%A, %d %B %Y")
    time_text = now.strftime("%I:%M:%S %p").lstrip("0")
    tz_text = now.tzname() or "Europe/London"
    return f"{date_text}, {time_text} ({tz_text})"
  except Exception as error:
    return json.dumps({"error": str(error)})

general_tools_list = [
  get_time,
]