#!/usr/bin/env bash
# Installs Tessera on a Linux machine with Tenstorrent Wormhole cards (n150,
# n300). Run it from a checkout of this repository:
#
#   scripts/install.sh            # install what is missing, asking first
#   scripts/install.sh --check    # only say what is there and what is missing
#   scripts/install.sh --system --test -y
#
# It checks the cards, their driver and hugepages; installs Go if needed;
# fetches and builds tt-metal v0.80.0, the version Tessera is built against
# (about 12 GB and an hour, the first time); builds tessc; and writes the
# environment that Tessera's programs need. Run again, it skips what is done.
#
# Options:
#   --check          report only; change nothing (exits 1 if something is missing)
#   --system         also install the system packages tt-metal needs, with
#                    tt-metal's own install_dependencies.sh (uses sudo)
#   --tt-metal DIR   tt-metal's checkout: used if it is built there, else cloned
#                    and built there (default: $TT_METAL_HOME, or ~/tt-metal-v0.80.0)
#   --bin DIR        where tessc goes (default: ~/.local/bin)
#   --env FILE       the environment file to write (default: ~/.config/tessera/env.sh)
#   --no-shell-rc    don't add the environment file to ~/.bashrc
#   --test           run scripts/card-test.sh quick on the card at the end
#   -y, --yes        don't ask before using sudo, downloading or building tt-metal
#
# The cards' driver (tt-kmd), firmware and hugepages are installed with
# Tenstorrent's tt-installer (https://github.com/tenstorrent/tt-installer);
# this script checks them and says what to do if they are missing.
set -euo pipefail

TT_METAL_VERSION=v0.80.0
TT_METAL_REPO=https://github.com/tenstorrent/tt-metal.git
TT_METAL_BUILD_FLAGS=${TT_METAL_BUILD_FLAGS:-}  # extra flags for build_metal.sh
GO_VERSION=${GO_VERSION:-1.27.4}                # installed if there is no Go 1.24 or newer
GO_MIN_MINOR=24
TESTED_KMD=2.5.0                                # what Tessera was tested with
TESTED_FIRMWARE=19.2.0

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CHECK=false SYSTEM=false TEST=false YES=false SHELL_RC=true
TT_METAL_DIR=${TT_METAL_HOME:-$HOME/tt-metal-$TT_METAL_VERSION}
BIN=$HOME/.local/bin
ENV_FILE=${XDG_CONFIG_HOME:-$HOME/.config}/tessera/env.sh

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case $1 in
    --check) CHECK=true ;;
    --system) SYSTEM=true ;;
    --test) TEST=true ;;
    -y | --yes) YES=true ;;
    --no-shell-rc) SHELL_RC=false ;;
    --tt-metal) TT_METAL_DIR=${2:?--tt-metal needs a directory}; shift ;;
    --bin) BIN=${2:?--bin needs a directory}; shift ;;
    --env) ENV_FILE=${2:?--env needs a file}; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done

missing=0
ok() { printf '  ok    %s\n' "$*"; }
note() { printf '  note  %s\n' "$*"; }
miss() { printf '  MISS  %s\n' "$*"; missing=$((missing + 1)); }
step() { printf '\n== %s\n' "$*"; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }
ask() {  # ask "question": true if yes (no, without a terminal to ask on, unless -y)
    $YES && return 0
    if [ ! -t 0 ]; then
        note "not asked, as there is no terminal (-y says yes): $1"
        return 1
    fi
    local answer
    read -r -p "  $1 [y/N] " answer
    case $answer in y | Y | yes | Yes) return 0 ;; esac
    return 1
}
have() { command -v "$1" > /dev/null 2>&1; }

# ---------------------------------------------------------------------------
step "The machine"
[ "$(uname -s)" = Linux ] || die "Tessera needs Linux; this is $(uname -s)"
[ "$(uname -m)" = x86_64 ] || die "tt-metal v0.80.0 is built for x86_64; this is $(uname -m)"
ok "Linux $(uname -r), x86_64"
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    case "${ID:-}:${VERSION_ID:-}" in
    ubuntu:22.04 | ubuntu:24.04) ok "$PRETTY_NAME" ;;
    *) note "${PRETTY_NAME:-this distribution}: tt-metal supports Ubuntu 22.04 and 24.04; other systems may need their packages installed by hand" ;;
    esac
fi

# ---------------------------------------------------------------------------
step "Tenstorrent cards"
wormholes=0 others=""
for dev in /sys/bus/pci/devices/*; do
    [ -r "$dev/vendor" ] && [ "$(cat "$dev/vendor")" = 0x1e52 ] || continue
    case $(cat "$dev/device") in
    0x401e) wormholes=$((wormholes + 1)) ;;
    0xb140) others="$others Blackhole" ;;
    0xfaca) others="$others Grayskull" ;;
    *) others="$others $(cat "$dev/device")" ;;
    esac
done
if [ "$wormholes" -gt 0 ]; then
    ok "$wormholes Wormhole card(s) on PCIe (an n300's second chip is reached through its first)"
else
    miss "no Wormhole card on PCIe: Tessera needs an n150 or n300"
fi
[ -z "$others" ] || note "also found:$others; Tessera has only been tested on Wormhole (Blackhole needs porting; tt-metal no longer supports Grayskull)"

kmd=$(cat /sys/module/tenstorrent/version 2> /dev/null || true)
if [ -n "$kmd" ] && ls /dev/tenstorrent/* > /dev/null 2>&1; then
    ok "driver tt-kmd $kmd, $(ls /dev/tenstorrent | wc -l) device(s) in /dev/tenstorrent (tested with $TESTED_KMD)"
else
    miss "the driver (tt-kmd) is not loaded: install it with Tenstorrent's tt-installer, then reboot:
          /bin/bash -c \"\$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)\"
          (it also installs the firmware, hugepages and tt-smi; Tessera was tested with firmware $TESTED_FIRMWARE)"
fi

# Without an IOMMU translating for the card (off, or iommu=pt), tt-metal
# needs a 1 GB hugepage per card for its command queues.
huge=$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2> /dev/null || echo 0)
if [ "$huge" -ge "$((wormholes > 0 ? wormholes : 1))" ] && grep -q ' /dev/hugepages-1G ' /proc/mounts; then
    ok "$huge hugepages of 1 GB, mounted at /dev/hugepages-1G"
else
    miss "1 GB hugepages: $huge reserved$(grep -q ' /dev/hugepages-1G ' /proc/mounts || echo ", /dev/hugepages-1G not mounted"); tt-installer sets them up (or: sudo $TT_METAL_DIR/install_dependencies.sh --hugepages)"
fi

card_users() { have lsof && lsof /dev/tenstorrent/* 2> /dev/null | awk 'NR > 1 {print $1 " (pid " $2 ")"}' | sort -u | tr '\n' ' '; }
busy=$(card_users || true)
[ -z "$busy" ] || note "the card is in use by: $busy; Tessera's programs need it to themselves"

# ---------------------------------------------------------------------------
step "Build tools"
tools_missing=""
for tool in git curl tar cmake ninja clang++-20 python3; do
    if have "$tool"; then
        ok "$tool: $("$tool" --version 2>&1 | head -1 | cut -c1-60)"
    else
        tools_missing="$tools_missing $tool"
    fi
done
if [ -n "$tools_missing" ]; then
    if $SYSTEM && ! $CHECK; then
        note "missing:$tools_missing; tt-metal's install_dependencies.sh installs them below"
    else
        miss "missing:$tools_missing; run with --system to install them with tt-metal's install_dependencies.sh (uses sudo)"
    fi
fi

# ---------------------------------------------------------------------------
step "Go"
GO=""
for g in "$(command -v go 2> /dev/null || true)" "$HOME/.local/go/bin/go" "$HOME/local/go/bin/go" /usr/local/go/bin/go; do
    [ -n "$g" ] && [ -x "$g" ] || continue
    minor=$("$g" env GOVERSION 2> /dev/null | sed -n 's/^go1\.\([0-9]*\).*/\1/p')
    if [ -n "$minor" ] && [ "$minor" -ge "$GO_MIN_MINOR" ]; then
        GO=$g
        break
    fi
done
if [ -n "$GO" ]; then
    ok "$("$GO" version) at $GO"
elif $CHECK; then
    miss "Go 1.$GO_MIN_MINOR or newer (the script installs Go $GO_VERSION in ~/.local/go)"
else
    [ ! -e "$HOME/.local/go" ] || die "$HOME/.local/go exists but holds no Go 1.$GO_MIN_MINOR or newer; move it aside and run again"
    ask "download Go $GO_VERSION (about 70 MB) from go.dev into ~/.local/go?" || die "Tessera needs Go 1.$GO_MIN_MINOR or newer"
    have curl && have python3 || die "installing Go needs curl and python3 (run with --system first)"
    tarball=go$GO_VERSION.linux-amd64.tar.gz
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "$tmp/$tarball" "https://go.dev/dl/$tarball"
    curl -fsSL -o "$tmp/releases.json" "https://go.dev/dl/?mode=json&include=all"
    want=$(python3 -c 'import json, sys
for r in json.load(open(sys.argv[1])):
    for f in r["files"]:
        if f["filename"] == sys.argv[2]:
            print(f["sha256"])' "$tmp/releases.json" "$tarball")
    got=$(sha256sum "$tmp/$tarball" | cut -d' ' -f1)
    [ -n "$want" ] && [ "$want" = "$got" ] || die "the Go download's checksum does not match go.dev's ($got, want ${want:-none})"
    mkdir -p "$HOME/.local"
    tar -C "$HOME/.local" -xzf "$tmp/$tarball"
    GO=$HOME/.local/go/bin/go
    ok "$("$GO" version) at $GO (checksum verified)"
fi

# ---------------------------------------------------------------------------
step "tt-metal $TT_METAL_VERSION"
built() { [ -f "$TT_METAL_DIR/build/lib/cmake/tt-metalium/tt-metalium-config.cmake" ]; }
if built; then
    version=$(git -C "$TT_METAL_DIR" describe --tags 2> /dev/null || echo unknown)
    if [ "$version" = "$TT_METAL_VERSION" ]; then
        ok "built at $TT_METAL_DIR"
    else
        note "built at $TT_METAL_DIR, but it is $version, not $TT_METAL_VERSION: Tessera may not build against it"
    fi
elif $CHECK; then
    miss "not built at $TT_METAL_DIR (the script clones and builds it there: about 12 GB and an hour)"
else
    if [ ! -d "$TT_METAL_DIR/.git" ]; then
        ask "clone tt-metal $TT_METAL_VERSION into $TT_METAL_DIR?" || die "Tessera needs tt-metal $TT_METAL_VERSION (or pass --tt-metal DIR)"
        have git || die "cloning tt-metal needs git (run with --system, or install git)"
        git clone --branch "$TT_METAL_VERSION" --depth 1 --recurse-submodules --shallow-submodules \
            "$TT_METAL_REPO" "$TT_METAL_DIR"
    fi
    if $SYSTEM; then
        ask "install tt-metal's system packages (clang 20, cmake, the SFPI kernel toolchain, ...) with sudo?" ||
            die "stopped before installing tt-metal's system packages"
        sudo "$TT_METAL_DIR/install_dependencies.sh"
    fi
    have clang++-20 || die "building tt-metal needs clang++-20: run again with --system"
    ask "build tt-metal now (about an hour)?" || die "stopped before building tt-metal"
    # shellcheck disable=SC2086
    (cd "$TT_METAL_DIR" && ./build_metal.sh --release $TT_METAL_BUILD_FLAGS)
    built || die "tt-metal's build finished without build/lib/cmake/tt-metalium"
    ok "built at $TT_METAL_DIR"
fi
if [ -d /opt/tenstorrent/sfpi ] || [ -d "$TT_METAL_DIR/runtime/sfpi" ]; then
    ok "the SFPI toolchain, which compiles kernels when programs run"
else
    miss "the SFPI toolchain (kernels are compiled with it at run time): run with --system"
fi

# ---------------------------------------------------------------------------
step "Tessera"
[ -f "$ROOT/go.mod" ] && grep -q '^module github.com/pavelgladyshev/tessera' "$ROOT/go.mod" ||
    die "run this script from a checkout of the Tessera repository"
if $CHECK; then
    if [ -x "$BIN/tessc" ]; then ok "tessc at $BIN/tessc"; else miss "tessc (the script builds it into $BIN)"; fi
    if [ -f "$ENV_FILE" ]; then ok "environment file $ENV_FILE"; else miss "environment file $ENV_FILE"; fi
else
    [ -n "$GO" ] || die "no Go to build tessc with"
    mkdir -p "$BIN"
    (cd "$ROOT" && "$GO" build -o "$BIN/tessc" ./cmd/tessc)
    ok "tessc built into $BIN/tessc"

    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" << EOF
# Tessera's environment, written by $ROOT/scripts/install.sh.
export TESSERA_HOME="$ROOT"
export TT_METAL_HOME="$TT_METAL_DIR"
export TT_METAL_RUNTIME_ROOT="$TT_METAL_DIR"
export TT_METAL_CACHE="\${XDG_CACHE_HOME:-\$HOME/.cache}/tessera/jit"  # tt-metal's compiled kernels
case ":\$PATH:" in *":$BIN:"*) ;; *) export PATH="$BIN:\$PATH" ;; esac
case ":\$PATH:" in *":$(dirname "$GO"):"*) ;; *) export PATH="$(dirname "$GO"):\$PATH" ;; esac
EOF
    ok "environment written to $ENV_FILE"
    line="[ -f \"$ENV_FILE\" ] && . \"$ENV_FILE\"  # Tessera"
    if ! $SHELL_RC; then
        note "not added to ~/.bashrc; run: . $ENV_FILE"
    elif grep -qF "$line" "$HOME/.bashrc" 2> /dev/null; then
        ok "~/.bashrc already reads it"
    elif ask "add a line to ~/.bashrc that reads it?"; then
        printf '\n%s\n' "$line" >> "$HOME/.bashrc"
        ok "~/.bashrc reads it from the next shell on"
    else
        note "not added to ~/.bashrc; run: . $ENV_FILE"
    fi
fi

# ---------------------------------------------------------------------------
if $TEST && ! $CHECK; then
    step "A first run on the card"
    busy=$(card_users || true)
    if [ "$missing" -gt 0 ]; then
        note "skipped: something above is missing"
    elif [ -n "$busy" ]; then
        note "skipped: the card is in use by $busy"
    else
        # shellcheck disable=SC1090
        . "$ENV_FILE"
        (cd "$ROOT" && scripts/card-test.sh quick) || die "the card test failed (its output is above)"
        ok "langcheck, limits and localstack give go run's results on the card"
    fi
fi

# ---------------------------------------------------------------------------
echo
if [ "$missing" -gt 0 ]; then
    echo "$missing thing(s) missing; see MISS above."
    exit 1
fi
if $CHECK; then
    echo "Everything Tessera needs is there."
else
    echo "Tessera is installed. In a new shell (or after: . $ENV_FILE), try:"
    echo "  cd $ROOT && tessc -o build/softplus ./examples/softplus && (cd build/softplus && ./build.sh)"
fi
