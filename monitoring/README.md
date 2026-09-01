# monitoring: CLI/TUI live dashboard (rich) and structured logging. No GUI.

## Resource readings

`monitoring/resources.py` reads CPU, memory and GPU, and `dae monitor` draws
them live.

    dae monitor                 live panel, until interrupted
    dae monitor --once          one frame
    dae monitor --json          one snapshot as JSON
    dae monitor --seconds 30    live, then stop

`snapshot()` returns a `ResourceSnapshot`, and `as_dict()` gives the stored
form for a log or a dashboard record.

The rule the module is built around: a value that could not be read comes back
as `None` and renders as "unavailable", never as zero. This machine's GPU is
the worked example, reporting power draw perfectly well while reporting its
power LIMIT as "[N/A]". A default of 0 W there would draw a power bar against
a limit of zero and read as a broken device rather than a missing field. A
measured zero, on the other hand, is a fact and survives.

Reading never influences. Nothing here sits on a compute path, and a failure
to read degrades to "unavailable" rather than raising into the run being
watched.
