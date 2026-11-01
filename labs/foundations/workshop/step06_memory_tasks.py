"""Step 6: a deterministic in-memory task state machine, before persistence."""
class MemoryTasks:
    def __init__(self):
        self._tasks = {}
        self._keys = {}
        self._next_id = 1

    def add(self, title, key):
        if not isinstance(title, str) or not title.strip():
            raise ValueError('non-empty title required')
        if not isinstance(key, str) or not key:
            raise ValueError('non-empty idempotency key required')
        title = title.strip()
        if key in self._keys:
            task = self._tasks[self._keys[key]]
            if task['title'] != title:
                raise ValueError('idempotency conflict')
            return dict(task), False
        task = {'id': self._next_id, 'title': title, 'done': False}
        self._tasks[task['id']] = task
        self._keys[key] = task['id']
        self._next_id += 1
        return dict(task), True

    def complete(self, task_id):
        if task_id not in self._tasks:
            raise KeyError(task_id)
        self._tasks[task_id]['done'] = True
        return dict(self._tasks[task_id])

    def list(self):
        return [dict(self._tasks[k]) for k in sorted(self._tasks)]

if __name__ == '__main__':
    store = MemoryTasks()
    print(store.add('Learn Python', 'one'))
    print(store.add('Learn Python', 'one'))
    print(store.complete(1))
    print(store.list())
