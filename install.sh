#!/usr/bin/env bash
# install.sh — Installs PTT STT GUI into the user's desktop environment.
#
# What this script does:
#   1. Copies the SVG icon into ~/.local/share/icons/hicolor/scalable/apps/
#   2. Writes a .desktop entry to ~/.local/share/applications/
#   3. Updates the icon/desktop caches
#   4. Verifies all required Python packages are present in ~/stt-env
#
# No sudo required.  Everything goes into XDG user directories.
#
# Usage:
#   bash install.sh [--uninstall]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ID="pl.tomw.PushToTalk"
VENV="${HOME}/stt-env"
MODEL_PATH="${HOME}/stt-models/polish"

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'
BOLD='\033[1m'

info()  { echo -e "${CYAN}[info]${RESET}  $*"; }
ok()    { echo -e "${GREEN}[ok]${RESET}    $*"; }
warn()  { echo -e "${YELLOW}[warn]${RESET}  $*"; }
error() { echo -e "${RED}[error]${RESET} $*" >&2; }
die()   { error "$*"; exit 1; }

# ── Uninstall ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--uninstall" ]]; then
    info "Uninstalling PTT STT GUI..."
    rm -f "${HOME}/.local/share/applications/${APP_ID}.desktop"
    rm -f "${HOME}/.local/share/icons/hicolor/scalable/apps/${APP_ID}.svg"
    xdg-mime default "" audio/x-wav 2>/dev/null || true
    gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
    update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
    ok "Uninstalled."
    exit 0
fi

echo -e "\n${BOLD}${CYAN}PTT STT — Installer${RESET}\n"

# ── Pre-flight checks ──────────────────────────────────────────────────────
info "Checking Python venv..."
[[ -f "${VENV}/bin/activate" ]] || die "venv not found at ${VENV}. Create it first:\n  python3 -m venv ~/stt-env && source ~/stt-env/bin/activate && pip install vosk sounddevice numpy evdev"

info "Checking VOSK model..."
[[ -d "${MODEL_PATH}" ]] || warn "Model not found at ${MODEL_PATH}. The app will show an error on startup. Download from https://alphacephei.com/vosk/models"

info "Checking required packages in venv..."
MISSING=()
for pkg in vosk sounddevice numpy; do
    "${VENV}/bin/pip" show "${pkg}" &>/dev/null || MISSING+=("${pkg}")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    warn "Missing packages: ${MISSING[*]}"
    info "Installing missing packages..."
    "${VENV}/bin/pip" install "${MISSING[@]}"
fi

info "Checking GTK4/Adwaita (system)..."
python3 -c "import gi; gi.require_version('Gtk','4.0'); gi.require_version('Adw','1'); from gi.repository import Gtk, Adw" 2>/dev/null \
    || die "GTK4 + libadwaita not available. Install:\n  sudo apt install python3-gi gir1.2-gtk-4.0 gir1.2-adw-1"

info "Checking wtype..."
command -v wtype &>/dev/null \
    || warn "wtype not found. 'Type to window' feature will not work.\n  Install: sudo apt install wtype"

# ── Directories ────────────────────────────────────────────────────────────
ICON_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
APPS_DIR="${HOME}/.local/share/applications"
mkdir -p "${ICON_DIR}" "${APPS_DIR}"

# ── Icon ───────────────────────────────────────────────────────────────────
info "Installing icon..."
cp "${SCRIPT_DIR}/assets/ptt-mic.svg" "${ICON_DIR}/${APP_ID}.svg"
ok "Icon installed: ${ICON_DIR}/${APP_ID}.svg"

# ── Desktop entry ─────────────────────────────────────────────────────────
info "Writing .desktop entry..."
LAUNCHER="${SCRIPT_DIR}/ptt-gui"
cat > "${APPS_DIR}/${APP_ID}.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Push-to-Talk STT
Name[pl]=Push-to-Talk STT
GenericName=Speech to Text
GenericName[pl]=Mowa na tekst
Comment=Polish speech recognition — push and hold to record
Comment[pl]=Rozpoznawanie mowy po polsku — przytrzymaj i mów
Exec=${LAUNCHER}
Icon=${APP_ID}
Terminal=false
Categories=AudioVideo;Audio;Accessibility;Utility;
Keywords=speech;stt;microphone;vosk;polish;ptt;
StartupNotify=true
StartupWMClass=${APP_ID}
DESKTOP
chmod +x "${APPS_DIR}/${APP_ID}.desktop"
ok "Desktop entry: ${APPS_DIR}/${APP_ID}.desktop"

# ── Refresh caches ────────────────────────────────────────────────────────
info "Refreshing icon cache..."
gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null && ok "Icon cache updated." || warn "gtk-update-icon-cache not found, skipping."

info "Refreshing desktop database..."
update-desktop-database "${APPS_DIR}" 2>/dev/null && ok "Desktop database updated." || warn "update-desktop-database not found, skipping."

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}Installation complete!${RESET}"
echo ""
echo "  Launch from GNOME Activities: search for 'Push-to-Talk'"
echo "  Launch from terminal:          ${LAUNCHER}"
echo "  CLI mode:                      ${SCRIPT_DIR}/ptt"
echo ""
echo "  To uninstall: bash ${SCRIPT_DIR}/install.sh --uninstall"
echo ""
