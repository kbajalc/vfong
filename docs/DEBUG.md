# Debugging Guide

This project mixes Python scripts, Cython extensions (`.pyx`), and C code
(the OSEA QRS detector, `vf_features_native.c`, and the WFDB library).
There are two independent debugging modes depending on what level you need
to inspect, plus a VS Code setup that gets very close to pure-Python comfort.

> **Note:** The original Cython implementation now lives in `hong/`. Run all
> `make …` targets and the `extract_one.py` / `feature_extraction.py` commands
> below from inside `hong/` (`cd hong`). The VS Code launch configs already set
> `cwd` to `hong/`.

---

## Quick reference

| Goal | Command | Tool |
|------|---------|------|
| Step through `.pyx` in VS Code / PyCharm / pdb | `make trace` then run normally | Python debugger |
| Step into C code (OSEA, vf_features_native.c) | `make debug` then `cygdb` or `gdb` | gdb / cygdb |
| Restore optimised build for real runs | `make release` | — |
| Clean all compiled artefacts | `make clean` | — |

The two build modes are independent — `make trace` is enough for 95 % of
debugging work. Only reach for `make debug` when you need to inspect the C
internals.

---

## Mode 1 — Python-level tracing (`make trace`)

### What it does

Compiles all Cython extensions with:

- C flags: `-O0 -g` (no optimisation, full debug symbols)
- Cython directive `linetrace=True` — emits a `__Pyx_TraceLine()` call before
  every Python-visible statement in the generated C code
- C macro `CYTHON_TRACE=1` and `CYTHON_TRACE_NOGIL=1` — activates those calls
  at compile time
- Cython directive `profile=True` — required for `linetrace` to work

The result is that `.pyx` files report line numbers to Python's `sys.settrace()`
hook, the same hook that `pdb`, VS Code's `debugpy`, and PyCharm's pydevd all
use. From the debugger's perspective the extensions look almost identical to
pure Python modules.

### Build

```bash
make trace
```

Run this once. The compiled `.so` files stay on disk; you do not need to set
any environment variable at runtime.

### Using pdb in the terminal

```bash
python -m pdb extract_one.py -r mitdb/100 -n 0
```

Useful pdb commands once inside a `.pyx` frame:

```
(Pdb) l          # list source lines around the current position
(Pdb) n          # next line (step over)
(Pdb) s          # step into a function call
(Pdb) p tcsc     # print a variable
(Pdb) b vf_features.pyx:595   # set a breakpoint by file and line
(Pdb) c          # continue to next breakpoint
```

### Using breakpoint() in source

You can drop `breakpoint()` anywhere in a `.py` or (with the trace build) in a
`.pyx` file and pdb drops into the interactive prompt at that line. Remove before
committing.

---

## Mode 2 — gdb / cygdb (`make debug`)

Use this when you need to step into C code: the OSEA QRS detector
(`osea/`), `vf_features_native.c` (LZ complexity), or the WFDB C library.

### What it does

Compiles all extensions with:

- C flags: `-O0 -g`
- `cythonize(..., gdb_debug=True)` — generates a `.pyx.gdb` XML file alongside
  each `.so`. These files teach gdb to map C stack frames back to Cython source
  lines.

### Build

```bash
make debug
```

### cygdb (Cython's gdb wrapper)

`cygdb` is installed as part of the `cython` package. It loads the `.pyx.gdb`
files automatically.

```bash
cygdb . -- --args python extract_one.py -r mitdb/100 -n 0
```

Inside cygdb:

```
(gdb) cy break vf_features.pyx:595   # Cython-aware breakpoint
(gdb) cy run
(gdb) cy next
(gdb) cy step                         # step into C / Cython calls
(gdb) cy print tcsc                   # print a Cython variable
(gdb) cy locals                       # show all local variables
(gdb) cy bt                           # Cython-aware backtrace
```

### Plain gdb

```bash
gdb --args python extract_one.py -r mitdb/100 -n 0
(gdb) run
# when stopped at a signal or crash:
(gdb) bt                              # C-level backtrace
(gdb) frame 3                         # switch to a specific frame
(gdb) info locals
```

### Restore after debugging

```bash
make release
```

Always restore before running `feature_extraction.py` on real data — the debug
build is 3–5× slower.

---

## VS Code setup for near-pure-Python debugging experience

With the right extensions and configuration, stepping through `.pyx` files in
VS Code feels almost identical to debugging `.py` files: breakpoints in the
gutter, variable inspection in the sidebar, call stack with `.pyx` filenames and
line numbers.

### Required extensions

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| **Python** | Microsoft | Core Python language support, `debugpy` integration |
| **Pylance** | Microsoft | Type checking and IntelliSense for `.py` files |
| **Cython+** | cython-collective | Syntax highlighting and basic IntelliSense for `.pyx` files |

Install all three from the Extensions panel (`Ctrl+Shift+X`) or:

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension cython-collective.cython-plus
```

**Cython+** is the actively maintained fork of the original Cython extension.
It gives `.pyx` files proper syntax colouring and makes them navigable. Without
it the files open as plain text and breakpoints in them are invisible.

### Recommended optional extensions

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| **C/C++** | Microsoft | Syntax and hover for `.c` / `.h` files in `osea/` |
| **GitLens** | GitKraken | Inline blame, useful when tracing which commit introduced a feature |
| **Error Lens** | Alexander | Inline error display, catches Python type errors without leaving the editor |

### Build step (one-time, or after changing `.pyx` files)

```bash
make trace
```

VS Code does not run `make trace` for you. You must rebuild whenever you edit a
`.pyx` file, otherwise the debugger steps through stale compiled code and line
numbers will be wrong.

### launch.json

`.vscode/launch.json` ships with these configurations. Hong's scripts live in `hong/`,
so `program` points there and `cwd` is set to `hong/` (so `file_lists/` and pyximport
resolve):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "extract_one: mitdb/100 seg 0",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/hong/extract_one.py",
            "args": ["-r", "mitdb/100", "-n", "0"],
            "cwd": "${workspaceFolder}/hong",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "extract_one: vfdb/422 begin 385788",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/hong/extract_one.py",
            "args": ["-r", "vfdb/422", "-b", "385788", "-s", "8"],
            "cwd": "${workspaceFolder}/hong",
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "feature_extraction (no parallelism)",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/hong/feature_extraction.py",
            "args": ["-o", "features/features_s8_debug.dat", "-s", "8", "-j", "1"],
            "cwd": "${workspaceFolder}/hong",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

**`justMyCode: false` is critical.** Without it VS Code skips all frames that
are not in your own source files, which includes every Cython extension. With it
set to `false` the debugger steps into `.pyx` frames normally.

### settings.json (workspace)

Create or update `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ],
    "files.associations": {
        "*.pyx": "cython",
        "*.pxd": "cython"
    },
    "editor.formatOnSave": false
}
```

The `files.associations` entry makes VS Code treat `.pyx` and `.pxd` files as
Cython regardless of the Cython+ extension's auto-detection. `extraPaths` helps
Pylance find the compiled extensions next to the source.

### Breakpoints in .pyx files

1. Open any `.pyx` file in VS Code (e.g. `vf_features.pyx`).
2. Click in the gutter to the left of a line number — a red dot appears.
3. Press `F5` to start the `extract_one` launch configuration.
4. The debugger pauses at your breakpoint inside the `.pyx` file.
5. The Variables panel shows Cython local variables; the Call Stack panel shows
   the full mixed Python/Cython stack.

### What still does not work

| Limitation | Workaround |
|------------|------------|
| `nogil` blocks are not traceable even in trace builds — `CYTHON_TRACE_NOGIL=1` is set but some compilers optimise the calls away | Add a `with gil:` wrapper around the specific line you want to inspect, rebuild |
| Cython typed `cdef` variables with no Python equivalent may show as `<optimized out>` in the Variables panel | Declare them as `object` temporarily during debugging |
| Editing a `.pyx` file does not trigger a rebuild — breakpoints silently shift | Run `make trace` after every `.pyx` edit; VS Code will reload the `.so` on next `F5` |
| Stepping into C functions (`lempel_ziv_complexity`, OSEA internals) is not possible in trace mode | Use `make debug` + gdb / cygdb for C-level inspection |

---

## Workflow summary

```
# First time setup
make trace                         # build with Python-level tracing
code .                             # open VS Code

# Open vf_features.pyx, set a breakpoint on a feature function
# Press F5 to launch "extract_one: mitdb/100 seg 0"
# Debugger pauses in .pyx — inspect variables, step line by line

# After editing any .pyx file
make trace                         # must rebuild to keep line numbers in sync

# When done debugging, restore speed for real data runs
make release
```
