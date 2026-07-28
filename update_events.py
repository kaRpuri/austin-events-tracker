"""Daily refresh for the Novara Austin/USA/Global events tracker.

Reads data.json, asks Claude (with the web_search tool) to find new or
updated events, and writes the result back to data.json. Run by
.github/workflows/update.yml on a daily cron; requires ANTHROPIC_API_KEY.
"""

import json
import os
import re
import sys
from datetime import date, timedelta

from anthropic import Anthropic

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = """You maintain a live events tracker for Kabir, founder at Novara Robotics \
(a physical-AI / robotics / hardware manufacturing startup based in Austin, TX). Today's date is {today}.

## Scope
Priority order: (1) Austin/Texas — the primary focus, most detail expected; (2) USA-wide — major \
national startup, robotics, manufacturing, and industrial-automation events worth traveling for; \
(3) Global — only genuinely flagship international events in the same space, not an exhaustive list.

Every event's `category` must be exactly one of: `community` (Austin startup/founder community), \
`manufacturing` (Austin-area manufacturing), `regional` (Texas but outside Austin), `national` \
(USA-wide, outside Texas), `global` (outside the USA).

## Current data
Here is the current tracker data as JSON (`events`: dated entries with fields id, title, org, \
category, tag, dateStart, dateEnd, time, location, cost, registerBy, link, desc; `watchlist`: \
recurring/ad-hoc entries with no confirmed date, fields title, org, cadence, meta, link):

{current_data}

## Task
Search the web for updates:
- Capital Factory's live calendar (capitalfactory.com/in-person) — FIESTA, Cup of Capital, Member \
Salon, AI Tinkerers, HACK AI, and any new founder/investor events.
- ARMA (arma-tx.org/events) — the 2026 State of Manufacturing Expo date once posted, and any newly \
listed luncheons/events.
- fedsupernova.com — schedule or registration changes.
- Austin-area robotics/hardware/physical-AI meetups (meetup.com/austin-robotics-ai, \
meetup.com/austin-hardware-startup, meetup.com/austin-hardware-meetup), HICAM (hicam.io), Prototown \
(prototown.com), Antler Austin, Q-Branch (q-branch.dev), Central Texas Angel Network \
(centraltexasangelnetwork.com / ctan.com) for newly confirmed dates or new sessions.
- General web search for newly announced Austin-area startup, venture, robotics, hardware, \
manufacturing, industrial-automation, or dual-use/defense-tech events in the next 6 months not \
already in the data.
- Newly announced or newly-dated USA-wide flagship manufacturing/robotics/automation conferences, \
and major startup ecosystem events, anywhere in the US.
- Newly announced or newly-dated global flagship events in robotics, physical AI, advanced \
manufacturing, or industrial automation — only major/flagship ones.

## Update rules
- For any `watchlist` item that now has a confirmed date, move it into `events` with a new stable \
`id` (kebab-case, e.g. event-name-YYYY-MM), filling in all fields you can find, with the correct \
`category`, then remove it from `watchlist`.
- For any genuinely new relevant event (check by title to avoid duplicates), add it to `events` (if \
dated) or `watchlist` (if not yet dated), with the correct `category`.
- For existing `events` entries, update fields if you find corrected info (date changes, \
registration deadlines, links) but NEVER change an existing event's `id` — the viewer's per-event \
"registered / attended / notes" marks in their browser are keyed to that id.
- Do not remove past events — the page automatically buckets them into history by date.
- If you're not confident about a date or detail, don't guess — leave it out.
- Set `lastUpdated` to "{last_updated_display}".

## Output
After you finish searching, respond with ONLY a single JSON object — no prose, no markdown code \
fences — matching this exact shape: {{"lastUpdated": "...", "events": [...], "watchlist": [...]}}. \
It must be the complete data (all unchanged entries included verbatim), not just a diff.
"""


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


def save_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output")
    return json.loads(text[start : end + 1])


def main():
    current = load_data()
    today = date.today()
    prompt = PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        current_data=json.dumps(current, indent=2),
        last_updated_display=today.strftime("%b %-d, %Y"),
    )

    client = Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 20}]
    messages = [{"role": "user", "content": prompt}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        tools=tools,
        messages=messages,
    )

    # Server-side tool loop can pause after many searches in one turn; resume it.
    hops = 0
    while response.stop_reason == "pause_turn" and hops < 5:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response.content},
        ]
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=tools,
            messages=messages,
        )
        hops += 1

    text_blocks = [b.text for b in response.content if b.type == "text"]
    if not text_blocks:
        print("No text block in model response; nothing to update.", file=sys.stderr)
        sys.exit(1)

    try:
        updated = extract_json(text_blocks[-1])
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Failed to parse model output as JSON: {e}", file=sys.stderr)
        print(text_blocks[-1], file=sys.stderr)
        sys.exit(1)

    if "events" not in updated or "watchlist" not in updated:
        print("Model output missing 'events' or 'watchlist' key; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)

    save_data(updated)
    print(f"Updated data.json: {len(updated['events'])} events, {len(updated['watchlist'])} watchlist items.")


if __name__ == "__main__":
    main()
