# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
from hashlib import sha256
from pathlib import Path
from subprocess import run
from sys import argv

ROOT = Path(__file__).parent.parent
SCRIPT = Path(__file__).relative_to(ROOT)
LOCKFILE = ROOT / "pdm.lock"
REQS = ROOT / "requirements.txt"

deps_hash = sha256(LOCKFILE.read_bytes()).hexdigest()[0:30]

if REQS.is_file():
  header = REQS.read_text().splitlines()[0]
  if deps_hash in header: exit(0)

if len(argv) > 1 and argv[1] == "--check":
  print(f"requirements.txt is outdated. Run `python {SCRIPT}` to update it.")
  exit(1)

print("Updating requirements.txt...")
header = f"# {deps_hash}"
deps = run(["pdm", "export", "--format", "requirements", "--without-hashes", "--group", ":all"], capture_output=True, cwd=ROOT, check=True).stdout.decode()

REQS_TMP = REQS.with_name(REQS.name + ".tmp")
REQS_TMP.write_text(header + "\n" + deps)
REQS_TMP.rename(REQS)
