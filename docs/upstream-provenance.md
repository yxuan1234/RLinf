# Upstream provenance

This LIBERO OPD release is based on [RLinf](https://github.com/RLinf/RLinf).
Its release branch starts from the original upstream commit:

`695680123732c14ee441a268d200176bd42a279e`

[fix(wrappers): ensure video flush for nested RecordVideo (#1279)](https://github.com/RLinf/RLinf/commit/695680123732c14ee441a268d200176bd42a279e)

The original upstream history and commit hashes are preserved. The release
changes are applied on top of that base, so they can be compared with the base
using `git diff 695680123732c14ee441a268d200176bd42a279e HEAD` after committing.
For a future upstream pull request, review this diff and rebase onto the current
upstream main branch as needed.

The release provides LIBERO online Student-rollout OPD, routed frozen teachers,
and optional same-state reference regularization. Offline teacher-observation
sampling, OPD reference replay, DAgger-CFM, and RoboTwin OPD recipes are excluded.
Private development branch history is not merged into the release branch.

RLinf's original license and copyright notices are retained.
