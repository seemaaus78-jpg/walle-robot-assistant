# WALL-E Hybrid Robot Assistant

An offline-first voice assistant for a small desktop robot, built around a
Radxa Cubie A7Z. It listens continuously, answers questions about cities,
translates phrases, and gestures with four servos — with no network connection
required for any of it.

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

## Why offline-first

A travel assistant is most useful exactly where connectivity is worst, and a
robot that behaves differently on the bench and in the field is a robot you
cannot debug. So the local models answer everything, and an online backend — if
you configure one — only ever gets first refusal. Anything it declines or fails
on falls through to the local path.

## Hardware

| Part | Role |
|---|---|
| Radxa Cubie A7Z | Compute (2 GB variant recommended) |
| INMP441 | I²S MEMS microphone |
| MAX98357A + 8 Ω speaker | I²S amplifier and audio out |
| 4 × SG90 | Neck pan/tilt, two arms |
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

python3 main_assistant.py
```

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
  assistant.py             routing and orchestration
  tts.py                   Piper invocation and playback
  stt.py                   PortAudio capture and Vosk decoding
  motion.py                50 Hz software PWM and gestures
scripts/
  fetch_models.sh          download Vosk, Piper and voices
  build_city_db.py         build world_cities.db from a GeoNames dump
systemd/                   unit file for running at boot
docs/                      hardware, architecture, setup, defects fixed
tests/                     103 tests, stdlib unittest, no hardware needed
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

103 tests, no dependencies beyond the standard library — so they also run on the
board itself, where installing pytest is a waste of a card you want for models.

They cover intent routing, city name extraction and lookup, the Piper command
construction, connectivity caching, servo pulse-width maths and the PWM thread,
config parsing, and the full response path with every device faked.

## Status and verification

The conversation logic is complete and tested. The hardware layer is written
against the datasheets and the design specification but has **not** been run on
a physical robot — I had no board, microphone, servos or speaker available.

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
[Argos Translate](https://github.com/argosopentech/argos-translate).
