# `remote.sh` — AI Agent Usage Guide

## Overview

`remote.sh` is a shell utility that lets you run code on a **remote HPC cluster** from your local machine. It handles three concerns automatically:

1. **File sync** — pushing your local project to the cluster and pulling results back
2. **Node allocation** — acquiring a GPU compute node via the SLURM job scheduler
3. **Remote execution** — running commands or background jobs on that node

As an agent, you will primarily use the `run` and `batch` subcommands to execute training scripts, evaluations, or data processing jobs. Do not run GPU jobs locally despite the alluring presence of an NVIDIA GPU.

---

## SLURM Primer for Agents

**SLURM** (Simple Linux Utility for Resource Management) is a job scheduler used on HPC clusters. Instead of SSH-ing directly into a powerful machine, you submit *jobs* to SLURM and it allocates resources (CPUs, GPUs, RAM) from a pool of compute *nodes*.

Key concepts you need:

| Concept | What it means in practice |
|---|---|
| **Node** | A physical machine in the cluster (e.g. `u22-04`) |
| **Allocation** | A reserved block of resources on a node, granted by SLURM |
| **`salloc`** | The SLURM command that requests an interactive allocation |
| **`squeue`** | Lists your currently running or queued jobs |
| **`scancel`** | Cancels a running job, freeing the allocation |
| **Idle node** | A node with no current jobs — the best target for a new allocation |

`remote.sh` manages all SLURM interactions for you. You do **not** need to call `salloc`, `squeue`, or `scancel` directly.

### How `remote.sh` acquires a node

When you run any command that requires a compute node (`run`, `shell`, `batch submit`), the script follows this resolution order:

```
1. Check local cache (.remote_jobs/.interactive_node)
   └─ still valid in squeue? → use it

2. Query squeue for any already-running interactive job
   └─ found? → cache it and use it

3. Scan for idle nodes matching NODE_FILTER (default: "u22")
   └─ try each idle node with salloc (16 GB RAM, 1 GPU, 6 h wall time)
      └─ poll for up to 90 s until allocation appears in squeue
```

This means **node acquisition is automatic and idempotent**: calling `run` twice will reuse the same allocation rather than requesting a new one.

---

## Quick Reference

```
remote.sh up                          # sync local → remote
remote.sh run <command>               # sync + run on compute node (blocking)
remote.sh down                        # pull results back (skips large model.pt)
remote.sh shell                       # interactive shell on compute node
remote.sh node status                 # show which node is currently allocated
remote.sh node stop                   # cancel allocation and clear cache
remote.sh batch submit <command>      # fire-and-forget background job
remote.sh batch list                  # list background jobs and their status
remote.sh batch stop <job_id>         # kill a specific background job
```

---

## Environment Variables

Override defaults by exporting these before calling `remote.sh`, or prefixing each call:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `ada` | SSH hostname of the cluster login node |
| `REMOTE_DIR` | `~/research/<cwd-basename>` | Project directory on the remote |
| `JOB_DIR` | `.remote_jobs` | Local and remote directory for job metadata |
| `INTERACTIVE_JOB_NAME` | `remote_dev` | SLURM job name for the interactive allocation |
| `LOCAL_HOME` | `$HOME` | Your local home directory (used for path rewriting) |
| `NODE_FILTER` | `u22` | Substring filter when scanning for idle nodes |

---

## Typical Agent Workflows

### 1. Run a one-off training script

```bash
./remote.sh run python train.py --epochs 10 --lr 1e-4
```

- Syncs local files to the remote first.
- Acquires a SLURM node if none is active.
- Executes the command **synchronously** — the call blocks until the command finishes.
- Use this for short jobs where you need the output immediately.

### 2. Submit a long training run in the background

```bash
./remote.sh batch submit python train.py --epochs 100 --lr 1e-4
```

- Returns a `job_id` (e.g. `20240315_142301_17382`) immediately.
- The job runs in the background on the compute node.
- stdout/stderr are written to `.remote_jobs/<job_id>.log` on the remote.

Check progress:

```bash
./remote.sh batch list
# JOB_ID                 STATUS   PID      COMMAND
# 20240315_142301_17382  running  84231    python train.py --epochs 100 --lr 1e-4
```

Cancel a job:

```bash
./remote.sh batch stop 20240315_142301_17382
```

### 3. Retrieve results after a job finishes

```bash
./remote.sh down
```

Syncs the remote `artifacts/` directory back to your local machine. Large files are excluded:
- `artifacts/**/model.pt` — intermediate model weights are skipped
- `artifacts/data/` — raw data is skipped

Only logs, metrics, configs, and final outputs are pulled.

### 4. Check what node is in use

```bash
./remote.sh node status
# Active node (cached): u22-07
```

### 5. Release the allocation when finished

```bash
./remote.sh node stop
```

This calls `scancel` on the SLURM job and clears the local cache. Do this when your work session is complete to free cluster resources for others.

---

## Files Written Locally

All agent-relevant state lives in `JOB_DIR` (`.remote_jobs/` by default):

| File | Purpose |
|---|---|
| `.remote_jobs/.interactive_node` | Cached name of the current compute node |
| `.remote_jobs/<job_id>.pid` | PID of the background process on the remote |
| `.remote_jobs/<job_id>.cmd` | The command that was submitted |
| `.remote_jobs/<job_id>.log` | stdout/stderr of the background job (on remote) |
| `.remote_jobs/<job_id>.start` | ISO timestamp of when the job was submitted |

---

## What Is and Isn't Synced

### `remote.sh up` excludes:
- `.git/` — version control history
- `__pycache__/`, `.venv/` — derived/environment files
- `artifacts/` — results live on the remote; don't overwrite them on upload
- `.remote_jobs/` — job metadata is managed separately

### `remote.sh down` includes only:
- Everything under `artifacts/` **except** `artifacts/**/model.pt` and `artifacts/data/`

If you need to pull a specific file not covered by `down`, open a shell and copy it manually:

```bash
./remote.sh shell
# then inside the shell:
cp artifacts/run_42/model.pt ~/local-backup/
```

---

## Error Handling & Diagnostics

| Symptom | Likely cause | What to do |
|---|---|---|
| `error: no idle nodes available` | All `u22` nodes are busy | Wait and retry, or change `NODE_FILTER` |
| `timed out waiting for allocation` | SLURM couldn't schedule within 90 s | Check cluster load; the allocation may still appear — run `node status` |
| `salloc launch failed` | SSH or SLURM config issue | Check `HOST` is reachable; inspect `.remote_jobs/.interactive.log` on the remote |
| `batch list` shows `stopped` unexpectedly | Process crashed or OOM-killed | Fetch `.remote_jobs/<job_id>.log` for the traceback |
| Garbled hostname errors | Old cached node name after a reallocation | Run `remote.sh node stop` to clear the cache, then retry |

---

## Constraints & Gotchas

- **Single quotes in commands break `batch submit`** — the command is passed via `echo '...'` to the remote shell. Wrap complex commands in a script file and submit the script instead:
  ```bash
  # Instead of:
  ./remote.sh batch submit python -c 'print("hello")'
  # Write a script and submit it:
  ./remote.sh batch submit bash run_experiment.sh
  ```

- **`run` is synchronous** — if your SSH connection drops, the remote process dies. For long jobs, always use `batch submit`.

- **One interactive allocation at a time** — `remote.sh` manages a single SLURM allocation per project directory. If you need parallel jobs, use `batch submit` multiple times within that allocation rather than acquiring multiple nodes.

- **6-hour wall time** — the interactive allocation expires after 6 hours. Any running batch jobs will be killed when it does. Plan long runs accordingly or re-acquire a node and resubmit.

- **Path rewriting** — your local `$HOME` paths are automatically rewritten to `$HOME` on the remote, so commands like `python $HOME/scripts/train.py` transfer correctly.
