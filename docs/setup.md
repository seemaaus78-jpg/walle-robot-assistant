# Setup

Target: Radxa Cubie A7Z running a headless Armbian or Radxa OS image.

Steps 1–3 happen on the board. Steps 4–6 are faster on a laptop; the results
copy across.

---

## 1. Enable I²S and check the GPIO chip

Nothing else works until the kernel exposes the I²S bus, and the two audio
devices will simply be absent rather than erroring.

Radxa images ship `rsetup`, which manages device tree overlays interactively:

```bash
sudo rsetup      # Overlays -> Manage overlays -> enable the I2S entry
sudo reboot
```

On a plain Armbian image, enable it in `/boot/armbianEnv.txt` instead and list
what your image actually provides:

```bash
ls /boot/dtb/*/overlay/ 2>/dev/null || ls /boot/dtbo/
```

> Overlay names differ between images and kernel versions for this board, and I
> have not verified them on a Cubie A7Z. Use the list your own image prints
> rather than copying a name from a forum post for a different Radxa model.

Confirm afterwards:

```bash
arecord -l          # should list the INMP441 capture device
aplay -l            # should list the MAX98357A playback device
gpioinfo            # note which gpiochip carries the 40-pin header
```

`gpioinfo` output is what goes into `config.toml` under `[motion]`. The chip
carrying the header is often **not** `gpiochip0`, and the `line` values are
offsets within that chip, not physical pin numbers.

Record a few seconds and play it back before going further:

```bash
arecord -D default -f S16_LE -r 16000 -c 1 -d 5 /tmp/test.wav
aplay /tmp/test.wav
```

If that round trip does not work, no amount of Python will help.

## 2. System packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev portaudio19-dev \
                    python3-libgpiod gpiod alsa-utils unzip curl
```

`python3-libgpiod` comes from apt rather than pip on purpose: the Python
bindings must match the libgpiod version the distribution ships, and the PyPI
package frequently does not. `walle/motion.py` supports both the v1 and v2
binding APIs, so either is fine.

## 3. Permissions

```bash
sudo usermod -aG gpio,audio "$USER"
# log out and back in for this to take effect
```

Running the assistant as root to avoid this works and is a bad habit; the
systemd unit uses `SupplementaryGroups=gpio audio` for the same reason.

## 4. The project and its dependencies

```bash
git clone https://github.com/seemaaus78-jpg/walle-robot-assistant.git \
    ~/travel_assistant
cd ~/travel_assistant

python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
```

`--no-cache-dir` matters here: pip's wheel cache will otherwise eat several
hundred megabytes of a card you want for models.

## 5. Models

```bash
./scripts/fetch_models.sh
```

That fetches the Vosk small English model, the Piper binary for your
architecture, and the English and Spanish voices — roughly 250 MB of downloads.

Then install the Argos language packs you actually want. Each pair is a separate
~100 MB download, so install only what you will use:

```bash
python3 - <<'EOF'
import argostranslate.package as pkg

pkg.update_package_index()
available = pkg.get_available_packages()
for from_code, to_code in [("en", "es")]:          # add pairs here
    match = next(
        p for p in available
        if p.from_code == from_code and p.to_code == to_code
    )
    print("installing", match)
    pkg.install_from_path(match.download())
EOF
```

## 6. City database

Build it on a laptop — indexing 200,000 rows on an SD card is slow — then copy
the finished file across.

```bash
curl -fLO https://download.geonames.org/export/dump/cities500.zip
curl -fLO https://download.geonames.org/export/dump/countryInfo.txt
curl -fLO https://download.geonames.org/export/dump/admin1CodesASCII.txt
unzip cities500.zip

python3 scripts/build_city_db.py cities500.txt \
    --countries countryInfo.txt \
    --admin1 admin1CodesASCII.txt \
    --aliases \
    -o world_cities.db

scp world_cities.db radxa@<robot-ip>:~/travel_assistant/
```

`cities500.txt` carries roughly 200,000 populated places. `cities1000.txt` is
about 140,000 and a smaller file if space is tight. `--aliases` also indexes
ASCII alternate names, so "bombay" finds Mumbai.

GeoNames data is CC BY 4.0 — keep the attribution if you redistribute the
database.

## 7. Configure

```bash
cp config.example.toml config.toml
```

Edit at minimum:

- `base_dir` — where you cloned to
- `[motion] chip` and each servo `line` — from your `gpioinfo` output
- `[audio] playback_device` — from `aplay -l`, if `default` is not the amplifier
- `[tts] binary` — the path `fetch_models.sh` printed

## 8. Try it without hardware first

```bash
python3 main_assistant.py --text \
    "tell me about tokyo" \
    "switch to translator" \
    "translate the museum is closed into french"
```

This runs the real intent routing, city lookup and translation, printing what
would be spoken instead of synthesising it. It is the fastest way to confirm the
models and database are wired up correctly before adding audio to the equation.

Then the real thing:

```bash
python3 main_assistant.py -v
```

## 9. Run at boot

```bash
sudo cp systemd/walle-assistant.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now walle-assistant
journalctl -u walle-assistant -f
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `arecord` gives silence | INMP441 `L/R` pin floating, or you are capturing the wrong channel. Tie it to GND. |
| No audio at all from the speaker | MAX98357A `SD` pin floating — it is a mode select, not just a shutdown. Strap it. |
| Speech plays too fast or too slow | Voice sample rate mismatch. The `.onnx.json` sidecar must sit beside its `.onnx`. |
| Robot answers its own voice | Acoustic coupling too strong. The mic is gated during playback, but move the mic away from the speaker or reduce volume. |
| Servos twitch but do not hold | 3.3 V signalling being marginal, or the servo rail dipping. See [hardware.md](hardware.md). |
| Board reboots when an arm moves | Servos sharing the logic rail. Give them their own boost converter. |
| `motion disabled (...)` in the log | Wrong `chip` name, or the user is not in the `gpio` group. |
| Everything is slow after boot | Argos paging in on a 1 GB board. Expected once; see [architecture.md](architecture.md). |
