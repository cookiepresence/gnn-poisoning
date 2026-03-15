#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-ada}"
REMOTE_DIR="${REMOTE_DIR:-~/research/$(basename "$(pwd)")}"
JOB_DIR="${JOB_DIR:-.remote_jobs}"
INTERACTIVE_JOB_NAME="${INTERACTIVE_JOB_NAME:-remote_dev}"
LOCAL_HOME="${LOCAL_HOME:-$HOME}"

# ── helpers ───────────────────────────────────────────────────────────────────

rewrite_home() {
  local cmd="$*" lh="${LOCAL_HOME%/}"
  [[ -z "$lh" ]] && { printf '%s' "$cmd"; return; }
  cmd="${cmd//${lh}\//'$HOME/'}"
  [[ "$cmd" == "$lh" ]] && cmd='$HOME'
  printf '%s' "$cmd"
}

# Run a command on HOST and return only its stdout, stripping any login banner
# that the remote shell prints before our command executes.
_ssh_capture() {
  local sentinel="###REMOTE_OUTPUT###"
  ssh "${HOST}" "unset HISTFILE; echo '${sentinel}'; $*" 2>/dev/null \
    | sed -n "/^${sentinel}\$/,\$ { /^${sentinel}\$/d; p }"
}

_idle_nodes() {
  _ssh_capture "/usr/sbin/pestat -G | grep idle | grep u22 | cut -d ' ' -f 1 | sort -rn"
}

# ── node management ───────────────────────────────────────────────────────────

_node_cache() { echo "${JOB_DIR}/.interactive_node"; }
_cache_node() { mkdir -p "${JOB_DIR}"; echo "$1" > "$(_node_cache)"; }
_query_node() { _ssh_capture "squeue --me --states=R -h -o '%N' 2>/dev/null | head -1" || true; }

_check_cache() {
  local f; f=$(_node_cache)
  [[ -f "$f" ]] || return 1
  local cached; cached=$(cat "$f")
  local found; found=$(_ssh_capture \
    "squeue --me --states=R -h -o '%N' 2>/dev/null | grep -Fx '${cached}'" || true)
  if [[ -n "$found" ]]; then echo "$cached"; return 0; fi
  rm -f "$f"; return 1
}

_start_on_node() {
  local node="$1"
  echo "[remote] Requesting Slurm allocation on ${node}..." >&2

  # Ensure the remote job dir exists before redirecting salloc log into it.
  ssh "${HOST}" "mkdir -p '${REMOTE_DIR}/${JOB_DIR}'" || true

  ssh "${HOST}" "
    nohup salloc \
      --job-name='${INTERACTIVE_JOB_NAME}' \
      --nodelist='${node}' \
      --mem=16G \
      --gres=gpu:1 \
      --time=6:00:00 \
      --no-shell \
      sleep infinity \
      </dev/null \
      >\"${REMOTE_DIR}/${JOB_DIR}/.interactive.log\" 2>&1 &
    disown \$!
  " # || { echo "[remote] error: salloc launch failed on ${HOST}" >&2; return 1; }

  local i n
  for (( i=0; i<45; i++ )); do
    sleep 2
    n=$(_query_node)
    if [[ -n "$n" ]]; then _cache_node "$n"; echo "$n"; return 0; fi
  done

  echo "[remote] error: timed out waiting for allocation on ${node} (90 s)" >&2
  ssh "${HOST}" "cat '${REMOTE_DIR}/${JOB_DIR}/.interactive.log'" 2>/dev/null || true
  return 1
}

ensure_node() {
  local node

  if node=$(_check_cache 2>/dev/null);        then echo "$node"; return 0; fi
  node=$(_query_node)
  if [[ -n "$node" ]];                        then _cache_node "$node"; echo "$node"; return 0; fi

  echo "[remote] No interactive job found. Searching for idle nodes..." >&2
  local idle_nodes
  idle_nodes=$(_idle_nodes)

  # FIX: was `echo $idle_nodes` (stdout), which polluted `node=$(ensure_node)`
  # captures and caused all subsequent SSH calls to fail with a garbled hostname.
  echo "[remote] Idle nodes: ${idle_nodes:-<none>}" >&2

  [[ -z "$idle_nodes" ]] && {
    echo "[remote] error: no idle nodes available (NODE_FILTER='${NODE_FILTER}')" >&2
    exit 1
  }

  for n in $idle_nodes; do
    if node=$(_start_on_node "$n"); then echo "$node"; return 0; fi
    echo "[remote] Trying next node..." >&2
  done

  echo "[remote] error: could not start interactive job on any idle node" >&2
  exit 1
}

node_status() {
  local node
  if   node=$(_check_cache 2>/dev/null);                  then echo "Active node (cached): ${node}"
  elif node=$(_query_node); [[ -n "$node" ]];             then _cache_node "$node"; echo "Active node (squeue): ${node}"
  else echo "No interactive job running."
  fi
}

node_stop() {
  local node job_id
  if node=$(_check_cache 2>/dev/null) || { node=$(_query_node); [[ -n "$node" ]]; }; then
    echo "[remote] Cancelling Slurm allocation (node: ${node})..." >&2
    job_id=$(ssh "${HOST}" \
      "squeue --me --states=R -h -o '%i' --name='${INTERACTIVE_JOB_NAME}' | head -1" 2>/dev/null || true)
    if [[ -n "$job_id" ]]; then
      ssh "${HOST}" "scancel '${job_id}'"
    else
      ssh "${HOST}" "squeue --me --states=R -h -o '%i' | xargs -r scancel" 2>/dev/null || true
    fi
    rm -f "$(_node_cache)"
    echo "Done."
  else
    echo "No interactive job to stop."
  fi
}

# ── sync ──────────────────────────────────────────────────────────────────────

sync_up() {
  rsync --verbose -az --delete --checksum \
    --exclude '.git/'        --exclude '__pycache__/' --exclude '.venv/' \
    --exclude '/artifacts/'  --exclude "${JOB_DIR}/" \
    -e "ssh" ./ "${HOST}:${REMOTE_DIR}/"
}

sync_down() {
  rsync -az --checksum -e "ssh" "${HOST}:${REMOTE_DIR}/" ./ \
    --exclude 'artifacts/**/model.pt' \
    --exclude 'artifacts/data/'       \
    --include 'artifacts/'            \
    --include 'artifacts/***'         \
    --exclude '*'
}

# ── remote execution ──────────────────────────────────────────────────────────

remote_run() {
  local node; node=$(ensure_node)
  ssh -o BatchMode=yes "${node}" zsh -s <<EOF
cd ${REMOTE_DIR}       
$(rewrite_home "$@")
EOF
}

node_shell() {
  local node; node=$(ensure_node)
  echo "[remote] Opening shell on ${node} (jump: ${HOST})..." >&2
  ssh -o BatchMode=yes "${node}" -t "cd ${REMOTE_DIR} && exec \$SHELL"
}

# ── batch ─────────────────────────────────────────────────────────────────────

batch_submit() {
  local cmd; cmd="$(rewrite_home "$@")"
  [[ -z "${cmd}" ]] && { echo "error: missing command for batch submit" >&2; exit 2; }

  # Note: commands containing single quotes will break the remote echo.
  # Wrap complex commands in a wrapper script instead.
  sync_up
  local node; node=$(ensure_node)
  ssh -o BatchMode=yes "${node}" "
    cd ${REMOTE_DIR}
    mkdir -p ${JOB_DIR}
    job_id=\$(date +%Y%m%d_%H%M%S)_\$RANDOM
    log=\"${JOB_DIR}/\${job_id}.log\"
    pidfile=\"${JOB_DIR}/\${job_id}.pid\"
    cmdfile=\"${JOB_DIR}/\${job_id}.cmd\"
    echo '${cmd}' > \"\${cmdfile}\"
    date -Iseconds > \"${JOB_DIR}/\${job_id}.start\"
    nohup bash \"\${cmdfile}\" > \"\${log}\" 2>&1 &
    echo \$! > \"\${pidfile}\"
    echo \"Submitted job: \${job_id}\"
  "
}

batch_list() {
  local node; node=$(ensure_node)
  ssh -o BatchMode=yes "${node}" "
    cd ${REMOTE_DIR}
    if [[ ! -d ${JOB_DIR} ]]; then echo 'No jobs found'; exit 0; fi
    shopt -s nullglob
    files=( ${JOB_DIR}/*.pid )
    (( \${#files[@]} == 0 )) && { echo 'No jobs found'; exit 0; }
    printf '%-22s %-8s %-8s %s\n' JOB_ID STATUS PID COMMAND
    for pidfile in \"\${files[@]}\"; do
      job_id=\$(basename \"\${pidfile}\" .pid)
      pid=\$(cat \"\${pidfile}\")
      cmd=\$(cat \"${JOB_DIR}/\${job_id}.cmd\" 2>/dev/null || echo '?')
      status=stopped
      ps -p \"\${pid}\" >/dev/null 2>&1 && status=running
      printf '%-22s %-8s %-8s %s\n' \"\${job_id}\" \"\${status}\" \"\${pid}\" \"\${cmd}\"
    done
  "
}

batch_stop() {
  local job_id="${1:-}"
  [[ -z "${job_id}" ]] && { echo "error: missing job_id" >&2; exit 2; }
  local node; node=$(ensure_node)
  ssh -o BatchMode=yes "${node}" "
    cd ${REMOTE_DIR}
    pidfile=\"${JOB_DIR}/${job_id}.pid\"
    [[ -f \"\${pidfile}\" ]] || { echo 'Job not found: ${job_id}'; exit 1; }
    pid=\$(cat \"\${pidfile}\")
    if ps -p \"\${pid}\" >/dev/null 2>&1; then
      kill \"\${pid}\"; sleep 1
      ps -p \"\${pid}\" >/dev/null 2>&1 && kill -9 \"\${pid}\"
    fi
    echo \"Stopped ${job_id} (pid \${pid})\"
  "
}

# ── CLI ───────────────────────────────────────────────────────────────────────

usage() {
  cat <<USAGE
Usage:
  remote.sh up                         sync local → remote (delta, checksum)
  remote.sh run <command...>           run command on compute node
  remote.sh down                       sync remote → local (skips model.pt)
  remote.sh shell                      interactive shell on compute node
  remote.sh node status                show active compute node
  remote.sh node stop                  cancel allocation & clear cache
  remote.sh batch submit <command...>  submit background job on compute node
  remote.sh batch list                 list background jobs
  remote.sh batch stop <job_id>        kill a background job

Env overrides:
  HOST=ada
  REMOTE_DIR=~/research/<project>
  JOB_DIR=.remote_jobs
  INTERACTIVE_JOB_NAME=remote_dev
  LOCAL_HOME=\$HOME
  SSH_CONFIG_FILE=~/.ssh/config
USAGE
}

case "${1:-}" in
  up)    shift; sync_up ;;
  run)   shift; sync_up; remote_run "$@" ;;
  down)  shift; sync_down ;;
  shell) shift; node_shell ;;
  node)
    shift
    case "${1:-}" in
      status) node_status ;;
      stop)   node_stop ;;
      *)      usage; exit 2 ;;
    esac
    ;;
  batch)
    shift
    case "${1:-}" in
      submit) shift; batch_submit "$@" ;;
      list)   batch_list ;;
      stop)   shift; batch_stop "$@" ;;
      *)      usage; exit 2 ;;
    esac
    ;;
  *) usage; exit 2 ;;
esac
