"""Step 7: demonstrate restart persistence in a temporary database."""
from pathlib import Path
from tempfile import TemporaryDirectory
from ..task_service import TaskStore

def restart_demo():
    with TemporaryDirectory() as directory:
        database = str(Path(directory) / 'tasks.sqlite3')
        first = TaskStore(database)
        try:
            task, _ = first.add('Persist this task', 'one')
            first.complete(task['id'])
        finally:
            first.close()
        second = TaskStore(database)
        try:
            return second.list()
        finally:
            second.close()

if __name__ == '__main__':
    print(restart_demo())
