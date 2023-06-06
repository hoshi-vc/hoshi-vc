# Copyright 2023 Hoshi-VC Developer <hoshi-vc@o137.net>.
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

# %%
import signal
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path

TASKS = Path(__file__).parent.parent / "tasks"
CURRENT = TASKS / "current"
DONE = TASKS / "done"

CURRENT.mkdir(parents=True, exist_ok=True)
DONE.mkdir(parents=True, exist_ok=True)

def execute_file(file: Path):
  choice = prompt_with_timeout(f"=== Proceed to run '{file.name}'?", 5, True)

  if not choice: exit(0)

  current_file = CURRENT / file.name
  file.rename(current_file)

  try:
    subprocess.run(["python", str(current_file)], check=True)
  except subprocess.CalledProcessError as e:
    print(f"=== Error on '{file.name}' : exit status {e.returncode}")

  time_str = time.strftime("%Y%m%dT%H%M%S")
  current_file.rename(DONE / (time_str + " " + file.name))

def watch_tasks():
  printed = False
  while True:
    files = list(TASKS.glob("*.py"))

    if not files:
      if not printed: print("=== Waiting for new tasks...")
      printed = True
      time.sleep(1)
      continue

    file = files[0]
    execute_file(file)
    printed = False

def getch():
  fd = sys.stdin.fileno()
  old_settings = termios.tcgetattr(fd)
  try:
    tty.setcbreak(fd)
    ch = sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
  return ch

def input_with_timeout(timeout: int):
  def handler(signum, frame):
    raise TimeoutError

  signal.signal(signal.SIGALRM, handler)
  signal.alarm(timeout)

  try:
    choice = getch()
  except TimeoutError:
    choice = None

  signal.alarm(0)
  return choice

def prompt_with_timeout(prompt: str, timeout: int, default: bool):
  try:
    t = 0
    while t < timeout:
      print(f"\r{prompt} [Y/n] ({timeout - t}s) ", end="", flush=True)
      choice = input_with_timeout(1)
      if choice is None:
        t += 1
      else:
        if choice.lower() == 'y': return True
        if choice.lower() == 'n': return False
    return default
  finally:
    print()

if __name__ == "__main__":
  watch_tasks()
