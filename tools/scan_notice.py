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

LICENSE_HASH = "3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04"
NOTICE = [
    "Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.",
    "This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.",
    "If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.",
]

EXCLUDED = [
    ".git/",
    ".husky/",
    "engine/fragment_vc/",
    "engine/hifi_gan/",
    "LICENSE",
]

EXCLUDED_EXT = [
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    ".json",
    ".lock",
    ".md",
    ".npmrc",
    ".png",
    ".prettierrc",
    ".toml",
    ".tool-versions",
    ".yaml",
]

if not (ROOT / "LICENSE").is_file():
  print("LICENSE is missing in the root directory")
  exit(1)

if sha256((ROOT / "LICENSE").read_bytes()).hexdigest() != LICENSE_HASH:
  print("LICENSE has been modified")
  exit(1)

ok = True

if len(argv) > 1:
  files = argv[1:]
else:
  files = run(['fdfind', '--type', 'file', "--hidden"], capture_output=True, cwd=ROOT).stdout.decode().splitlines()

for file in files:
  file = str((ROOT / file).relative_to(ROOT))

  if any([file.startswith(excluded) for excluded in EXCLUDED]): continue
  if any([file.endswith(ext) for ext in EXCLUDED_EXT]): continue

  contents = (ROOT / file).read_text()
  if all([line in contents for line in NOTICE]): continue

  print(f"Missing notice in {file}")
  ok = False

if not ok:
  print(f"Please add the notice to the files above or add them to EXCLUDED in {SCRIPT}")
  exit(1)
