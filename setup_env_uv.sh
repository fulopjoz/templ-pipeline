#!/usr/bin/env bash
# setup_env_uv.sh – one-liner bootstrap for TEMPL Pipeline
# Usage:
#   curl -sL https://raw.githubusercontent.com/<ORG>/<REPO>/setup_env_uv.sh | bash
# or, after cloning the repo, simply run:
#   ./setup_env_uv.sh
# The script will:
#   1. Install uv (if missing)
#   2. Create .venv/ with uv venv
#   3. Install templ_pipeline with all extras
#   4. Drop into an activated interactive shell

set -euo pipefail
IFS=$'\n\t'

#------------- helper logging -------------------------------------------------
CLR_RESET="\033[0m"
CLR_GREEN="\033[32m"
CLR_CYAN="\033[36m"
CLR_RED="\033[31m"
info()  { echo -e "${CLR_CYAN}[INFO]${CLR_RESET} $*"; }
ok()    { echo -e "${CLR_GREEN}[ OK ]${CLR_RESET} $*"; }
err()   { echo -e "${CLR_RED}[ERR]${CLR_RESET} $*" >&2; }

abort() { err "$1"; exit 1; }

#------------- python version check -------------------------------------------
PY_MIN=3.9
PY_VER=$(python3 -c 'import sys,platform; print(platform.python_version())' 2>/dev/null || true)
[[ -z "$PY_VER" ]] && abort "Python 3 not found. Please install Python >=${PY_MIN}."
vercomp() { [[ "$1" == "$2" ]] && return 0; local IFS=.; local i ver1=($1) ver2=($2); for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do ver1[i]=0; done; for ((i=0; i<${#ver1[@]}; i++)); do [[ -z ${ver2[i]} ]] && ver2[i]=0; if ((10#${ver1[i]} > 10#${ver2[i]})); then return 0; elif ((10#${ver1[i]} < 10#${ver2[i]})); then return 1; fi; done; return 0; }
vercomp "$PY_VER" "$PY_MIN" || abort "Python ${PY_MIN}+ required (found ${PY_VER})."

#------------- uv installation -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  info "Installing uv package manager…"
  curl -Ls https://astral.sh/uv/install.sh | bash
  # uv installs to ~/.cargo/bin – add if not present
  export PATH="$HOME/.cargo/bin:$PATH"
  ok "uv installed"
else
  ok "uv already present"
fi

#------------- virtual environment -------------------------------------------
ENV_DIR=".venv"
if [[ ! -d "$ENV_DIR" ]]; then
  info "Creating virtual environment in $ENV_DIR/ …"
  uv venv "$ENV_DIR"
else
  info "$ENV_DIR/ already exists – reusing"
fi

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

#------------- dependency installation ----------------------------------------
EXTRAS="all"
if [[ ${1:-} == "--dev" ]]; then
  EXTRAS="all,dev"
  shift
fi
info "Installing templ_pipeline[${EXTRAS}] … this may take a minute"
uv pip install -e "./templ_pipeline[${EXTRAS}]" "$@"
ok "templ_pipeline installed"

#------------- finish ---------------------------------------------------------
cat <<EOF
${CLR_GREEN}✅  Environment ready.${CLR_RESET}
To activate later:   source ${ENV_DIR}/bin/activate
Running interactive shell inside the environment now. Exit with Ctrl-D.
EOF

exec "$SHELL" -i 