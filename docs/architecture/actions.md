# Actions

`qml_observer.actions` implements the blueprint's four-level intervention
model (plan.md §7): log-only, warn, pause (not yet distinct from warn --
see below), and stop, plus a `"adaptive"` mode for opting into stopping on
a `degraded` diagnosis.

- **`Action`** (`actions/base.py`) -- shared interface: `execute(diagnosis)
  -> ActionResult`. Every built-in action catches its own internal errors,
  so a broken custom `Action` still can't crash a training loop (the
  monitor also catches at a higher level as a second line of defense).
- **`LogAction`** -- always executes, never raises; records every
  diagnosis regardless of severity.
- **`AlertAction`** -- terminal + logger warning for any non-`"info"`
  severity diagnosis.
- **`StopAction`** -- records a stop request via a `.triggered` flag for
  the caller's own training loop to check (`monitor.should_stop()`); never
  reaches into the loop directly, preserving the non-invasive core
  principle.
- **`PauseAction`** -- not yet implemented (Milestone 13); `"pause"` mode
  currently behaves identically to `"warn"`, a deliberate conservative
  choice rather than a silent no-op.

**`ActionPolicy`** (`actions/policies.py`) selects which `Action` to run
for a given `DiagnosisResult` and mode. It enforces the addendum §1
degraded-diagnosis safety rule: a `degraded=True` diagnosis never selects
`StopAction` unless `mode="adaptive"` **and** the caller explicitly passed
`allow_stop_on_degraded=True`.
