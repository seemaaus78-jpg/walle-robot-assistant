#!/usr/bin/env bash
# Download the offline models the assistant needs.
#
#     ./scripts/fetch_models.sh [target-dir]
#
# Roughly 250 MB of downloads and about 400 MB on disk once unpacked. Run it
# with the SD card mounted and plenty of free space.

set -euo pipefail

TARGET="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STT_DIR="$TARGET/models_stt"
TTS_DIR="$TARGET/models_tts"
BIN_DIR="$TARGET/bin"

VOSK_MODEL="vosk-model-small-en-us-0.15"
VOSK_URL="https://alphacephei.com/vosk/models/${VOSK_MODEL}.zip"

PIPER_VERSION="2023.11.14-2"
ARCH="$(uname -m)"
case "$ARCH" in
    aarch64|arm64) PIPER_ARCH="linux_aarch64" ;;
    armv7l)        PIPER_ARCH="linux_armv7l" ;;
    x86_64)        PIPER_ARCH="linux_x86_64" ;;
    *) echo "unsupported architecture: $ARCH" >&2; exit 1 ;;
esac
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"

VOICES_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"
# Each voice needs both the .onnx and its .onnx.json sidecar: the sidecar
# carries the sample rate, and guessing it plays the voice at the wrong pitch.
VOICES=(
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx"
    "es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json"
)

need() {
    command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }
}
need curl
need unzip
need tar

mkdir -p "$STT_DIR" "$TTS_DIR" "$BIN_DIR"

echo "==> Vosk speech model"
if [ -d "$STT_DIR/$VOSK_MODEL" ]; then
    echo "    already present, skipping"
else
    curl -fL --retry 3 -o "$STT_DIR/model.zip" "$VOSK_URL"
    unzip -q "$STT_DIR/model.zip" -d "$STT_DIR"
    rm -f "$STT_DIR/model.zip"
fi

echo "==> Piper binary ($PIPER_ARCH)"
if [ -x "$BIN_DIR/piper/piper" ]; then
    echo "    already present, skipping"
else
    curl -fL --retry 3 -o "$BIN_DIR/piper.tar.gz" "$PIPER_URL"
    tar -xzf "$BIN_DIR/piper.tar.gz" -C "$BIN_DIR"
    rm -f "$BIN_DIR/piper.tar.gz"
fi

echo "==> Piper voices"
for voice in "${VOICES[@]}"; do
    name="$(basename "$voice")"
    if [ -f "$TTS_DIR/$name" ]; then
        echo "    $name already present, skipping"
        continue
    fi
    echo "    $name"
    curl -fL --retry 3 -o "$TTS_DIR/$name" "$VOICES_BASE/$voice"
done

cat <<EOF

Done.

  Vosk model : $STT_DIR/$VOSK_MODEL
  Piper      : $BIN_DIR/piper/piper
  Voices     : $TTS_DIR

Point config.toml at the Piper binary you just downloaded:

    [tts]
    binary = "$BIN_DIR/piper/piper"

Still to do: build the city database (scripts/build_city_db.py) and install the
Argos language packs (see docs/setup.md).
EOF
