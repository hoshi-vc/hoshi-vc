# Simple file-based task queue

Related: [tools/task_queue.py](../tools/task_queue.py)

## Usage

1. Run `python tools/task_queue.py`.
2. Copy the Python script you want to run into this folder.

## Notes

- Files are run in order of their names.
- Files will be moved to `tasks/current` before execution.
  - Be careful when using `__file__` for example.
- Files will be moved to `tasks/done` after execution.
