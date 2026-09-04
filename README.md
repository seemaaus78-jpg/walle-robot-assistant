# WALL-E Hybrid Robot Assistant

A hybrid voice assistant for a small desktop robot, built around a Radxa Cubie
A7Z. It listens continuously, answers questions about cities, translates
phrases, and gestures with four servos.

**Two answering paths, one robot.** With Wi-Fi and a free Gemini API key it
answers from the cloud. Without either — no signal, no key, rate limited, API
down — the on-board Vosk, SQLite and Argos models answer instead, and the robot
behaves identically apart from less detail. Nothing about it *requires* a
network.

Design inspiration: <https://youtu.be/5MZ6O6yT73M>

Example session (illustrative — populations come from whichever GeoNames
extract you build):

```
  "tell me about kyoto"       ->  Kyoto is in Kyoto, Japan. It has a
                                  population of roughly 1,463,723 people.

  "switch to translator"      ->  Switched to translator mode.   [waves]

  "translate the museum is    ->  el museo esta cerrado
   closed into spanish"

  "nod"                       ->  [nods]
```

Everything above runs from the SD card. Wi-Fi, when present, is an optional
accelerator — never a requirement.

## How the two paths divide

| | Offline | Online (Gemini) |
|---|---|---|
| City questions | Where it is, population, region — fixed fields from GeoNames | Anything: what to eat, when to go, whether it is worth the detour |
| Translation | One installed Argos pair, ~100 MB each on disk | Any language pair, nothing stored locally |
| Latency | Tens of milliseconds | A network round trip, capped at 6 s |
| Available | Always | With Wi-Fi, a key, and quota left |

A travel assistant is most useful exactly where connectivity is worst, and a
robot that behaves differently on the bench and in the field is one you cannot
debug. So the local path is the guaranteed one and the cloud only ever gets
first refusal — anything it declines, times out on, or errors on falls straight
through.

Two rules keep that honest:

- **Motion and shutdown never leave the robot.** Only questions go upstream.
  "wave", "nod" and "shut down" are handled locally whatever the network is
  doing; a robot that stops obeying you because an API is slow is a worse robot.
- **Failure is always downward.** Nothing in `walle/online.py` can raise into
  the main loop. A rate limit or server error also puts the cloud path to sleep
  for a minute rather than retrying into the limit. The only symptom you ever
  see is a less detailed answer.

## Hardware

| Part | Role |
|---|---|
| Radxa Cubie A7Z | Compute (2 GB variant recommended) |
| INMP441 | I²S MEMS microphone |
| MAX98357A + 8 Ω speaker | I²S amplifier and audio out |
| 4 × SG90 | Neck pan/tilt, two arms |
| 2 × DC gear motors + mini L298N | The tracks |
| 3.7 V 6000 mAh LiPo + TP4056 + 5 V boost | Power |

Wiring, pin straps, the servo signalling caveats and the power design are in
**[docs/hardware.md](docs/hardware.md)**. Read the power section before
soldering — there are two things about the TP4056 that will bite you.

## Software stack

| Layer | Component |
|---|---|
| Speech to text | Vosk, small English model, offline |
| Text to speech | Piper, offline neural voices |
| Translation | Argos Translate, offline |
| Cloud answering | Gemini API, optional, stdlib `urllib` only |
| City knowledge | SQLite over a GeoNames extract (~200k places) |
| Motion | libgpiod, 50 Hz software PWM |
| OS | Headless Armbian / Radxa OS |

## Quick start

Full instructions, including enabling I²S and finding your GPIO offsets, are in
**[docs/setup.md](docs/setup.md)**. The short version, on the board:

```bash
git clone https://github.com/seemaaus78-jpg/walle-robot-assistant.git ~/travel_assistant
cd ~/travel_assistant
python3 -m venv .venv && source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt

./scripts/fetch_models.sh          # Vosk model, Piper binary, voices
cp config.example.toml config.toml # then edit the GPIO lines and audio device

# Optional: free key from https://aistudio.google.com/apikey
export GEMINI_API_KEY=your-key-here

python3 main_assistant.py
```

The startup log tells you which mode you are in:

```
walle.online: online backend: Gemini (gemini-2.5-flash)     <- hybrid
walle.online: no API key in $GEMINI_API_KEY; running offline-only
```

The key is read from the environment, never from `config.toml`, so it cannot be
committed alongside your GPIO offsets. `--offline` forces the local path even
with a key and a connection, which is the easiest way to hear the difference.

### Try it with no hardware at all

You do not need the robot, a microphone, a speaker or any models to exercise the
conversation logic:

```bash
python3 main_assistant.py --text \
    "tell me about tokyo" \
    "switch to translator" \
    "translate the museum is closed into french" \
    "wave your hand"
```

This runs the real intent routing, city lookup and translation and prints what
would be spoken. It is the fastest way to check a change before copying it to
the SD card.

## What it responds to

| Say | It does |
|---|---|
| "go forward" / "turn left" / "stop" | Drives the tracks — "stop" bypasses everything |
| "what do you see" | Takes one photo and describes it |
| "read this" | Reads the text in a photo — a menu, a sign, a label |
| "what does this sign say in english" | Reads it and translates it |
| "travel guide for kyoto" | Fetches a full guide online and keeps it on the card |
| "what are the restaurants in kyoto" | Reads that section back, online or off |
| "what have you saved" | Lists the cities it has guides for |
| "forget kyoto" / "delete all guides" | Deletes saved guides |
| "tell me about *city*", "how big is *city*", "where is *city*" | Looks the place up and describes it |
| "switch to translator" / "switch to city guide" | Changes mode, waves |
| "translate *phrase*" | Translates into the current target language |
| "translate *phrase* into french" | Translates into that language, just this once |
| "set language to german" | Changes the default target language |
| "wave", "nod" | Gestures |
| "are you online", "system status" | Reports connectivity, mode and language |
| "what can you do" | Lists its commands |
| "shut down", "go to sleep" | Rests the servos and exits cleanly |

Command phrasing lives in `walle/intents.py` and is plain data — add your own
without touching the rest.

## What it sees

A camera makes it answer the questions a traveller actually points a lens at: a
view, a menu, a sign in a script they cannot read.

```
"what do you see"                     → A narrow street of wooden houses,
                                        lanterns strung between the balconies.
"read this"                           → RAMEN 850 YEN, GYOZA 400 YEN
"what does this menu say in spanish"  → Ramen 850 yenes. Gyoza 400 yenes.
```

Two rules, both deliberate:

**It captures only when asked.** No background loop, no preview stream, no
motion trigger. It also says *"Let me look"* aloud before the shutter — a camera
that fires silently is a worse object to have on a desk than one that announces
itself. Nothing is stored unless `save_captures` is turned on, which it is not
by default.

**Vision is online-only, and says so.** There is no room on a 1 GB board for a
vision model beside speech, translation and synthesis, so with no connection it
answers *"I need an internet connection to see"* rather than guessing — and does
not open the shutter at all.

## Guides it keeps

Asking about a city while online fetches a full travel guide — areas worth
walking, where to eat, stations and how to get around, where to stay, emergency
numbers and hospitals — and writes it to the card. Ask again with no signal and
it is still there, along with the map tiles for that city.

Every answer says how old the guide is (*"this guide was saved 3 weeks ago"*),
because a restaurant list is a snapshot rather than a fact. Anything past
`refresh_after_days` is read back with an explicit out-of-date warning.

The emergency section is handled differently. A language model can invent a
plausible hospital address as easily as recall a real one, and acting on a wrong
one matters in a way that a wrong restaurant does not — so that section is
always read with a caution attached, telling you to check it against a local
source.

Guides live in `travel_guides.db`, separate from `world_cities.db`. Deleting
everything you saved is one file, and it can never damage the reference data the
robot needs to work. Manage them by voice, or with `scripts/guides.py`:

```bash
python3 scripts/guides.py list          # what is saved, and how old
python3 scripts/guides.py show kyoto    # print one in full
python3 scripts/guides.py delete kyoto  # forget one city
python3 scripts/guides.py clear         # forget everything (asks first)
python3 scripts/guides.py size          # how much space it takes
```

## How it fits together

```
  INMP441 --> Vosk --> intents.parse --> +--> cities.py  ---+
                                         |                  |
                                         +--> translation --+--> Piper --> MAX98357A
                                         |                  |
                                         +--> motion.py  ---+
```

The split runs along one line: everything that decides *what to say* is
separable from everything that touches a device. That is why the decision layer
is fully covered by tests with no hardware attached.

More detail, including the memory budget, is in
**[docs/architecture.md](docs/architecture.md)**.

## Layout

```
main_assistant.py          entry point; --text runs without hardware
config.example.toml        copy to config.toml and edit for your board
walle/
  config.py                TOML config, defaults, path resolution
  intents.py               transcript -> intent (pure functions)
  cities.py                name normalisation, n-gram scan, SQLite lookup
  translation.py           Argos wrapper with cached language pairs
  net.py                   cached connectivity probe
  online.py                Gemini backend, rate-limit cooldown
  guides.py                travel guides cached on the card
  camera.py                single-still capture, on request only
  drive.py                 two tracks through an H-bridge
  assistant.py             routing and orchestration
  tts.py                   Piper invocation and playback
  stt.py                   PortAudio capture and Vosk decoding
  motion.py                50 Hz software PWM and gestures
scripts/
  fetch_models.sh          download Vosk, Piper and voices
  build_city_db.py         build world_cities.db from a GeoNames dump
systemd/                   unit file for running at boot
docs/                      hardware, architecture, setup, defects fixed
tests/                     372 tests, stdlib unittest, no hardware needed
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

372 tests, no dependencies beyond the standard library — so they also run on the
board itself, where installing pytest is a waste of a card you want for models.

They cover intent routing, city name extraction and lookup, the Piper command
construction, connectivity caching, servo pulse-width maths and the PWM thread,
config parsing, and the full response path with every device faked.

The Gemini backend is tested through a fake transport, so no test touches the
network. That covers the parts that actually matter in the field: that a rate
limit hands back to SQLite, that gestures still work with a dead API, that a
safety block or a malformed response produces a local answer rather than
silence, and that the API key travels in a header and never in a URL.

## Status and verification

The conversation logic, both answering paths and the handoff between them are
complete and tested. The hardware layer is written against the datasheets and
the design specification but has **not** been run on a physical robot — I had no
board, microphone, servos or speaker available.

The Gemini backend has not been run against the live API either: it is tested
against a fake transport, so the request shape and every failure path are
covered, but the first real call is yours to make.

Specifically unverified: the software PWM timing in practice, the RAM figures,
the 4.5–5.5 hour battery estimate, the device tree overlay procedure, and Vosk
and Piper's real-time performance on the A7Z. Treat those as things to measure
on the bench.

`main_assistant.py` was written from a draft supplied with the specification.
Sixteen defects in that draft — including a shell injection on spoken text, a
robot that fell silent whenever it had Wi-Fi, and servos driven with a static
high line instead of PWM — are catalogued with their fixes and tests in
**[docs/defects-fixed.md](docs/defects-fixed.md)**.

## Attribution

City data from [GeoNames](https://www.geonames.org/), CC BY 4.0. Speech
recognition by [Vosk](https://alphacephei.com/vosk/), synthesis by
[Piper](https://github.com/rhasspy/piper), translation by
[Argos Translate](https://github.com/argosopentech/argos-translate), cloud
answering by the [Gemini API](https://ai.google.dev/).
