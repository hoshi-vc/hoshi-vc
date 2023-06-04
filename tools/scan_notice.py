# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
from pathlib import Path
from subprocess import run

ROOT = Path(__file__).parent.parent
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

ok = True

files = run(['fdfind', '--type', 'file', "--hidden"], capture_output=True, cwd=ROOT).stdout.decode().splitlines()
for file in files:
  if any([file.startswith(excluded) for excluded in EXCLUDED]): continue
  if any([file.endswith(ext) for ext in EXCLUDED_EXT]): continue

  contents = (ROOT / file).read_text()
  if all([line in contents for line in NOTICE]): continue

  print(f"Missing notice in {file}")
  ok = False

if not ok:
  print("Please add the notice to the files above or add them to EXCLUDED in scan.py")
  exit(1)
