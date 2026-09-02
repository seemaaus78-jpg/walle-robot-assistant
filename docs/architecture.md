# Architecture

## Operational workflow

```
                    +------------------------+
                    |      Robot boots       |
                    +-----------+------------+
                                v
                    +------------------------+
                    |  Load offline models   |
                    |  Vosk / Argos / SQLite |
                    +-----------+------------+
                                v
                    +------------------------+
                    |   Continuous listening |
                    |      (INMP441 mic)     |
                    +-----------+------------+
                                v
                    +------------------------+
                    |  Vosk: speech -> text  |
                    +-----------+------------+
                                v
                        Parse intent (walle/intents.py)
                                |
     +--------------+-----------+-----------+---------------+
     v              v                       v               v
[ COMMAND ]   [ MODE SWITCH ]         [ CITY QUERY ]  [ TRANSLATION ]
 wave / nod    city <-> translate      n-gram scan     Argos en->xx
 shutdown      set language            SQLite lookup
     |              |                       |               |
     +--------------+-----------+-----------+---------------+
                                v
                    Online backend configured
                    and reachable? -- yes --> try it,
                                |             fall through on
                                no            failure or decline
                                v
                    +------------------------+
                    |   Piper text-to-speech |
                    +-----------+------------+
                                v
                    +------------------------+
                    |  Speaker (MAX98357A)   |
                    |  mic gated while       |
                    |  speaking              |
                    +------------------------+
```

## Connectivity is an accelerator, never a gate

The offline pipeline answers every supported question on its own. Wi-Fi is only
consulted if an online backend has been configured, and anything that backend
declines or fails to answer falls through to the local models.

This is a deliberate inversion of the usual "online first" shape, for two
reasons. A travel assistant is most useful exactly where connectivity is worst.
And a robot whose behaviour differs between the bench and the field is a robot
you cannot debug.

No online backend ships with this repository. `walle/assistant.py` defines the
`OnlineBackend` protocol — one method, `answer(intent) -> Reply | None` — and
returning `None` means "I have nothing, use the local models".

## Module map

| Module | Responsibility | Hardware needed to test |
|---|---|---|
| `walle/config.py` | TOML config, defaults, path resolution | none |
| `walle/intents.py` | Transcript → intent. Pure functions. | none |
| `walle/cities.py` | Name normalisation, n-gram scan, SQLite lookup | none |
| `walle/translation.py` | Argos wrapper, cached language pairs | Argos packs |
| `walle/net.py` | Cached connectivity probe | none |
| `walle/assistant.py` | Routing, response text, orchestration | none |
| `walle/tts.py` | Piper invocation, playback | speaker |
| `walle/stt.py` | PortAudio capture, Vosk decoding | microphone |
| `walle/motion.py` | 50 Hz software PWM, gestures | servos |

The split is along one line: everything that decides *what to say* is separable
from everything that touches a device. That is why `python3 -m unittest` covers
the decision layer completely with no hardware attached, and why
`--text` can exercise a full conversation on a laptop.

## Memory budget

Design targets for the 1 GB board:

| Component | Target RSS |
|---|---|
| Linux headless (Armbian / Radxa OS, no desktop) | ~120 MB |
| Vosk small English model | ~100 MB |
| Argos Translate, one language pair loaded | ~220 MB |
| Piper, during synthesis | ~130 MB |
| SQLite page cache for `world_cities.db` | ~30 MB |
| Python runtime, PortAudio, gpiod | ~20 MB |
| **Total** | **~620 MB** |

Two honest caveats:

- These are the design targets from the specification, not measurements. I have
  not run this on the board. Measure with
  `systemd-cgtop` or `ps -o rss= -p $(pgrep -f main_assistant)` before relying
  on them.
- 620 MB of 1024 MB leaves very little headroom once the page cache, the ALSA
  buffers and a `journalctl` session are accounted for. On a 1 GB board, enable
  zram and expect the first translation after boot to be slow while Argos pages
  its model in. **The 2 GB variant removes the problem entirely and is worth the
  price difference.**

Argos is the largest single consumer, and it is only needed in translator mode.
If memory is tight, the cheapest win is to stop calling `ArgosTranslator.warm_up()`
at boot and let the model load on first use.
