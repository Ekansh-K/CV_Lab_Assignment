#!/usr/bin/env bash
# Run a remote bash snippet on Thunder instance 0 via `tnr connect 0` only.
set -euo pipefail
export PATH="/home/muthibazz/.tnr/bin:$PATH"
export COLUMNS=200
export LINES=50

if [[ $# -lt 1 ]]; then
  echo "usage: $0 '<remote bash>'" >&2
  exit 2
fi

REMOTE_CMD=$1
TIMEOUT=${REMOTE_TIMEOUT:-180}

# Marker so we can strip the MOTD/banner.
payload=$(cat <<EOF
stty cols 200 rows 50 2>/dev/null || true
echo __TNR_BEGIN__
${REMOTE_CMD}
echo __TNR_END__:\$?
exit
EOF
)

tmp=$(mktemp)
set +e
printf '%s\n' "$payload" | timeout "$TIMEOUT" tnr connect 0 >"$tmp" 2>&1
rc=$?
set -e

python3 - <<'PY' "$tmp" "$rc"
import sys, re
path, outer = sys.argv[1], int(sys.argv[2])
text = open(path, errors="replace").read()
# strip ANSI
text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
text = text.replace("\r", "")
begin = text.find("__TNR_BEGIN__")
end = text.rfind("__TNR_END__")
if begin == -1:
    sys.stderr.write(text[-4000:] + "\n")
    sys.stderr.write("tnr connect: no __TNR_BEGIN__ marker\n")
    sys.exit(outer or 1)
body = text[begin + len("__TNR_BEGIN__"):]
if end != -1:
    chunk = text[end:]
    m = re.search(r"__TNR_END__:(\d+)", chunk)
    inner = int(m.group(1)) if m else 1
    body = text[begin + len("__TNR_BEGIN__"):end]
else:
    inner = outer or 1
sys.stdout.write(body.strip() + "\n")
sys.exit(inner if begin != -1 else (outer or 1))
PY
rm -f "$tmp"
