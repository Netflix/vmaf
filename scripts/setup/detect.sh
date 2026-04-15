#!/usr/bin/env bash
# Dispatcher: detect host OS/distro and run the matching setup script.
# Idempotent. Safe to re-run. Never sudo-escalates without user input.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$OSTYPE" == "darwin"* ]]; then
  exec "$DIR/macos.sh" "$@"
fi

if [[ -n "${MSYSTEM:-}" || "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
  echo "Windows detected — use scripts\\setup\\windows.ps1 from PowerShell."
  exit 2
fi

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
  echo "Unsupported OSTYPE: $OSTYPE"
  exit 2
fi

# Linux: read /etc/os-release.
if [[ ! -f /etc/os-release ]]; then
  echo "Cannot detect Linux distro — /etc/os-release missing."
  exit 2
fi

# shellcheck disable=SC1091
. /etc/os-release

case "$ID" in
  ubuntu | debian | linuxmint | pop)
    exec "$DIR/ubuntu.sh" "$@"
    ;;
  arch | manjaro | cachyos | endeavouros)
    exec "$DIR/arch.sh" "$@"
    ;;
  fedora | rhel | rocky | alma)
    exec "$DIR/fedora.sh" "$@"
    ;;
  alpine)
    exec "$DIR/alpine.sh" "$@"
    ;;
  *)
    # Fallback: check ID_LIKE.
    for like in ${ID_LIKE:-}; do
      case "$like" in
        debian) exec "$DIR/ubuntu.sh" "$@" ;;
        arch)   exec "$DIR/arch.sh"   "$@" ;;
        fedora | rhel) exec "$DIR/fedora.sh" "$@" ;;
      esac
    done
    echo "Unsupported distro: $ID ($PRETTY_NAME)"
    echo "Supported: Ubuntu/Debian/Mint/Pop, Arch/Manjaro/CachyOS/EndeavourOS,"
    echo "           Fedora/RHEL/Rocky/Alma, Alpine, macOS, Windows."
    exit 2
    ;;
esac
