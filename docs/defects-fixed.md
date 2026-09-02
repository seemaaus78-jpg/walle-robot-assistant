# What changed from the draft script, and why

The specification for this robot came with a working outline of
`main_assistant.py`. The structure and the feature set here are that outline;
the differences are the defects found while turning it into something that runs
unattended on a battery.

Each entry says what the draft did, what goes wrong, and where the fix is
tested. Nothing below has been verified on the physical robot — see
[Verification status](#verification-status).

## Summary

| # | Defect | Severity |
|---|---|---|
| 1 | Online path answered nothing — robot fell silent on Wi-Fi | Critical |
| 2 | Spoken text was interpolated into a shell command | Critical |
| 3 | Servos driven with a static high line instead of PWM | Critical |
| 4 | No cleanup: servos left powered, audio stream leaked | High |
| 5 | One bad utterance killed the whole loop | High |
| 6 | Robot transcribed and answered its own voice | High |
| 7 | City name taken as the last word only | High |
| 8 | `NULL` population crashed the answer | High |
| 9 | Failed translation returned English, spoken in a foreign voice | High |
| 10 | Playback sample rate hard-coded to 22050 Hz | Medium |
| 11 | Voice filename derived as `{lang}_US.onnx` | Medium |
| 12 | Connectivity probed over HTTPS, per utterance, blocking | Medium |
| 13 | Capture buffer size disagreed with the read size | Medium |
| 14 | `description` column assumed to exist | Medium |
| 15 | No accent or case folding on city lookups | Medium |
| 16 | City database opened read-write | Low |

---

## 1. The online path answered nothing

```python
if is_connected():
    print("Online Mode Active")     # and nothing else
else:
    ...                             # the entire question-answering pipeline
```

Every answer lived in the `else`. With Wi-Fi available the robot listened,
transcribed, printed one line to a console nobody was watching, and said
nothing — failing in exactly the condition a user would assume made it work
better, and working on a bench with the Wi-Fi off.

**Now:** the offline pipeline is the guaranteed path. An online backend is
optional, gets first refusal when configured and reachable, and anything it
declines or throws on falls through to the local models.
`walle/assistant.py:respond`.

**Tested:** `tests/test_assistant.py::OnlineRegressionTests` — including that
the answer is byte-identical online and offline.

## 2. Spoken text was interpolated into a shell command

```python
cmd = f'echo "{text}" | piper --model ... | aplay ...'
os.system(cmd)
```

`text` here is a machine transcription of whatever was said near the robot, or a
`description` field out of a downloaded database. A double quote truncates the
sentence; a backtick or `$(...)` executes. This is a shell injection whose input
is "sound in the room".

It also breaks ordinary content: any city description containing an apostrophe
or a quotation mark mangles the utterance.

**Now:** `subprocess.Popen` with argument lists, text written to Piper's stdin,
no shell anywhere. `walle/tts.py`.

**Tested:** `tests/test_tts.py::CommandTests::test_text_never_reaches_the_argument_vector`.

## 3. Servos driven with a static high line

```python
def wave_hand():
    os.system("gpioset gpiochip1 15=1")   # Turn Servo ON
    pass
```

An SG90 is not an on/off actuator. It reads a 50 Hz pulse train and positions
its horn by pulse *width*. A line held high is not a valid servo signal, so the
horn either sits still or drifts against its end stop, drawing stall current
from a rail the board is sharing. The line was also never brought low and the
horn never returned to rest, so the first wave left an arm out and a servo
buzzing until power was cut.

**Now:** a real 50 Hz pulse train from a single background thread, angles
clamped to each servo's configured mechanical range, and automatic detach once a
pose has settled. `walle/motion.py`.

**Tested:** `tests/test_motion.py` — pulse-width maths, clamping, that commands
produce rising and falling edges, and that the servo goes slack after the hold
window.

**Caveat:** software PWM from CPython jitters by a few hundred microseconds.
Fine for gestures, not for steady holds. See [hardware.md](hardware.md).

## 4. No cleanup path

The draft had no signal handling and no teardown. `Ctrl-C` or a `systemctl stop`
left the PortAudio stream open, the GPIO lines claimed, and the servos powered.

**Now:** `SIGINT`/`SIGTERM` handled, components closed in reverse acquisition
order, and teardown tolerates a component that raises so one stuck device cannot
block the rest. `main_assistant.py`, `walle/assistant.py:close`.

**Tested:** `tests/test_assistant.py::RunLoopTests::test_close_tolerates_components_that_raise`,
`tests/test_motion.py::PwmThreadTests::test_close_is_idempotent`.

## 5. One bad utterance killed the loop

Any exception inside `while True` propagated out of `main()`. A single
malformed database row or a transient audio error ended the session, and on a
robot with no screen the failure is indistinguishable from a flat battery.

**Now:** each utterance is handled in isolation; a failure is logged, apologised
for out loud, and the loop continues.

**Tested:** `tests/test_assistant.py::RunLoopTests::test_a_failing_utterance_does_not_kill_the_loop`.

## 6. The robot answered itself

Continuous capture with no gate means the microphone hears Piper through the
speaker, Vosk transcribes it, and the assistant answers its own sentence —
which it then hears again.

**Now:** capture is drained but discarded while the speaker is active, and the
recogniser is reset afterwards so no fragment of the robot's own voice survives
into the next utterance. `walle/stt.py`, gated on `PiperSpeaker.is_speaking`.

## 7. City name taken as the last word

```python
words = text.split()
city = words[-1]
```

"tell me about new york" searched for `york`. "how big is san francisco"
searched for `francisco`. Any two-word city was unreachable.

**Now:** n-grams across the whole utterance, longest and leftmost first, after
stripping carrier phrases like "tell me about". Runs made entirely of stop-words
are dropped before they reach SQLite. `walle/cities.py:candidate_names`.

**Tested:** `tests/test_cities.py::CandidateTests`, `::LookupTests::test_multi_word_city_wins_over_its_suffix`.

## 8. `NULL` population crashed the answer

```python
return f"{city_name} is in {country}. It has a population of roughly {pop:,} people."
```

`f"{None:,}"` raises `TypeError`. GeoNames records 0 or nothing for a large
number of small places, so this turned a missing data point into a crash — and,
before fix #5, into a dead robot.

**Now:** the population sentence is omitted when the figure is missing, and the
builder normalises GeoNames' "0 means unknown" to `NULL` so the robot never
announces a population of zero. `walle/cities.py:City.summary`.

**Tested:** `tests/test_cities.py::LookupTests::test_null_population_does_not_crash`.

## 9. Failed translation returned the input

```python
except Exception:
    return text
```

The caller then handed that untranslated English to a Spanish Piper voice. A
missing language pack produced confident-sounding nonsense instead of an error —
the worst failure mode for a translation device, because the user cannot tell it
failed.

**Now:** failures raise `TranslationUnavailable`, and the assistant says, in
English, that the language pack is not installed. `walle/translation.py`.

**Tested:** `tests/test_assistant.py::TranslationFailureTests`.

## 10. Playback rate hard-coded

`aplay -r 22050` was fixed regardless of voice. Piper ships voices at 16000 Hz
(`low`), 22050 Hz (`medium`) and 24000 Hz (`high`); a `high` voice played at
22050 Hz runs about 9% slow and low-pitched.

**Now:** read from the voice's `.onnx.json` sidecar, with the configured
fallback only if that is unreadable. `walle/tts.py:voice_sample_rate`.

**Tested:** `tests/test_tts.py::SampleRateTests`.

## 11. Voice filename derived by string formatting

`f"models_tts/{lang}_US.onnx"` produces `es_US.onnx` for Spanish — a file that
does not exist for any Piper voice. Piper names encode a full locale and a
quality tier (`es_ES-davefx-medium`), which cannot be reconstructed from a bare
language code.

**Now:** voices are mapped explicitly per language in config, and a missing one
is reported rather than guessed. `walle/config.py`, `walle/tts.py:_voice_for`.

**Tested:** `tests/test_tts.py::VoiceSelectionTests`, `tests/test_config.py::DefaultsTests::test_spanish_voice_is_a_real_piper_name`.

## 12. Connectivity probed over HTTPS, per utterance

```python
urllib.request.urlopen('https://1.1.1.1', timeout=2)
```

This completes a TLS handshake against a bare IP address. Certificate validation
against an IP frequently fails on a healthy connection, so the robot decides it
is offline while it is not. It also put a blocking round trip of up to two
seconds between hearing a sentence and starting to answer it, on every
utterance.

**Now:** a plain TCP connect to `1.1.1.1:53`, cached for 30 seconds.
`walle/net.py`.

**Tested:** `tests/test_net.py::CachingTests` with a fake clock.

## 13. Capture buffer disagreed with the read size

The stream was opened with `frames_per_buffer=8000` and read `4000` frames per
call. The ring buffer filled faster than it drained, and PortAudio dropped audio
— typically the front of an utterance, which is where the command word is.

**Now:** one configured value feeds both. `walle/config.py:AudioConfig.block_frames`.

**Tested:** `tests/test_config.py::DefaultsTests::test_capture_block_matches_the_stream_buffer`.

## 14. `description` column assumed to exist

`SELECT country, population, description FROM cities` fails outright if the
database was built without that column — which every GeoNames-derived build is,
since GeoNames has no descriptions.

**Now:** columns are detected once at open; optional ones are included only if
present, and required ones missing produce a clear rebuild instruction rather
than an `OperationalError` mid-sentence. `walle/cities.py:_detect_columns`.

**Tested:** `tests/test_cities.py::SchemaTolerganceTests`.

## 15. No accent or case folding

`WHERE LOWER(name)=LOWER(?)` does not match "bogota" against "Bogotá", and Vosk
emits unaccented lower-case text. Every accented city name was unreachable.

**Now:** an indexed `name_norm` column holds an accent-stripped, case-folded,
punctuation-free form, and lookups normalise the query the same way.

**Tested:** `tests/test_cities.py::NormaliseTests`, `::LookupTests::test_accent_folded_lookup`.

## 16. Database opened read-write

`sqlite3.connect(DB_PATH)` opens for writing and creates the file if absent, so
a typo in the path silently produced an empty database that answered "not
found" forever, and a bug could write to it.

**Now:** opened `mode=ro` via URI, and a missing file raises immediately with
the path in the message.

**Tested:** `tests/test_cities.py::LookupTests::test_database_is_opened_read_only`.

---

## Verification status

**Verified here:** everything with a test reference above. 103 tests,
`python3 -m unittest discover -s tests -t .`, all passing on CPython 3.11.

**Not verified:** anything touching hardware. No Radxa board, INMP441,
MAX98357A, SG90 or LiPo was involved in writing this. In particular the
following are reasoned from datasheets and the specification, not measured:

- that the software PWM timing is good enough for your servos in practice
- the RAM figures in [architecture.md](architecture.md)
- the 4.5–5.5 hour battery estimate
- the device tree overlay procedure in [setup.md](setup.md)
- that Piper and Vosk perform acceptably on the A7Z's CPU

Treat all of those as claims to test on the bench, not as results.
