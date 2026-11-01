"""Reference-counted paged token metadata with transactional append and COW.

Payloads are token IDs standing in for computed KV rows. This module does not
store neural-network K/V tensors or execute PagedAttention.
"""
from __future__ import annotations
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
import heapq


class OutOfPages(RuntimeError):
    """An append cannot reserve every required page; the table is unchanged."""


@dataclass
class Page:
    data: list[int] = field(default_factory=list)
    refs: int = 0


class PagePool:
    def __init__(self, pages: int = 32, block_size: int = 4):
        if type(pages) is not int or type(block_size) is not int or pages < 1 or block_size < 1:
            raise ValueError("pages and block_size must be positive")
        self.block_size = block_size
        self.pages = [Page() for _ in range(pages)]
        self.free = list(range(pages))
        self.tables: dict[str, list[int]] = {}

    def create(self, rid: str) -> None:
        if not isinstance(rid, str) or not rid:
            raise ValueError("nonempty string table ID required")
        if rid in self.tables:
            raise ValueError("duplicate table ID")
        self.tables[rid] = []

    def tokens(self, rid: str) -> list[int]:
        return [t for bid in self.tables[rid] for t in self.pages[bid].data]

    def _counts(self, bids) -> Counter:
        ids = tuple(bids)
        if any(type(bid) is not int or not 0 <= bid < len(self.pages) for bid in ids):
            raise ValueError("page IDs must be valid nonnegative integers")
        return Counter(ids)

    def retain(self, bids: list[int] | tuple[int, ...]) -> None:
        counts = self._counts(bids)
        # Validate the whole operation before changing even the first refcount.
        if any(self.pages[bid].refs <= 0 for bid in counts):
            raise AssertionError("cannot retain a free page")
        for bid, count in counts.items():
            self.pages[bid].refs += count

    def unpin(self, bids: list[int] | tuple[int, ...]) -> None:
        counts = self._counts(bids)
        if any(self.pages[bid].refs < count for bid, count in counts.items()):
            raise AssertionError("double free or insufficient references")
        for bid, count in counts.items():
            page = self.pages[bid]
            page.refs -= count
            if page.refs == 0:
                page.data.clear()
                heapq.heappush(self.free, bid)

    def release(self, rid: str) -> None:
        self.unpin(self.tables.pop(rid))

    def fork(self, parent: str, child: str) -> None:
        # A missing parent or invalid child must not leave an empty table behind.
        if parent not in self.tables:
            raise KeyError(parent)
        if not isinstance(child, str) or not child or child in self.tables:
            raise ValueError("child requires a new nonempty string ID")
        bids = self.tables[parent].copy()
        self.retain(bids)
        self.tables[child] = bids

    def adopt(self, rid: str, bids: tuple[int, ...]) -> None:
        if self.tables[rid]:
            raise ValueError("adopt requires an empty table")
        if len(set(bids)) != len(bids):
            raise ValueError("a request table cannot repeat a physical page")
        self.retain(bids)
        self.tables[rid] = list(bids)

    def needed(self, rid: str, count: int) -> int:
        if type(count) is not int or count < 0:
            raise ValueError("count must be a nonnegative integer")
        if count == 0:
            return 0
        table = self.tables[rid]
        size = sum(len(self.pages[bid].data) for bid in table)
        total = (size + count + self.block_size - 1) // self.block_size
        copy_tail = bool(table and size % self.block_size and self.pages[table[-1]].refs > 1)
        return total - len(table) + int(copy_tail)

    def append(self, rid: str, values: list[int]) -> None:
        if any(type(x) is not int or x < 0 for x in values):
            raise ValueError("token IDs must be nonnegative integers")
        count = self.needed(rid, len(values))
        if count > len(self.free):
            raise OutOfPages(f"need {count} free pages; have {len(self.free)}")
        # Reserve before touching an existing page: failed allocations are atomic.
        reserved = [heapq.heappop(self.free) for _ in range(count)]
        for bid in reserved:
            self.pages[bid].refs = 1
        available = iter(reserved)
        table = self.tables[rid]
        if values and table:
            tail = self.pages[table[-1]]
            if len(tail.data) < self.block_size and tail.refs > 1:
                old = table[-1]
                new = next(available)
                self.pages[new].data = tail.data.copy()
                table[-1] = new
                self.unpin([old])
        for token in values:
            if not table or len(self.pages[table[-1]].data) == self.block_size:
                table.append(next(available))
            self.pages[table[-1]].data.append(token)
        assert list(available) == [], "reservation arithmetic is inconsistent"

    def check(self, cache_refs: Counter | None = None) -> None:
        expected = Counter(bid for table in self.tables.values() for bid in table)
        expected.update(cache_refs or {})
        assert len(self.free) == len(set(self.free)), "duplicate free-list entry"
        for bid, page in enumerate(self.pages):
            assert page.refs == expected[bid], (bid, page.refs, expected[bid])
            assert (bid in self.free) == (page.refs == 0)
            assert len(page.data) <= self.block_size
            assert page.refs > 0 or not page.data
        for table in self.tables.values():
            assert len(table) == len(set(table)), "same page repeated in a request"
            assert all(len(self.pages[bid].data) == self.block_size for bid in table[:-1])


class PrefixCache:
    """Exact full-page prefixes. Namespace includes model/adapter/tenant identity.

    Keeping the whole token tuple as a key avoids hashing/collision machinery in
    the teaching implementation. Real engines use more compact representations.
    """
    def __init__(self, pool: PagePool, max_entries: int = 16):
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("max_entries must be nonnegative")
        self.pool = pool
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple[str, tuple[int, ...]], tuple[int, ...]] = OrderedDict()

    def refs(self) -> Counter:
        return Counter(bid for bids in self.entries.values() for bid in bids)

    def publish(self, rid: str, namespace: str, max_tokens: int) -> int:
        if type(max_tokens) is not int or max_tokens < 0:
            raise ValueError("max_tokens must be a nonnegative integer")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("nonempty cache namespace required")
        if self.max_entries == 0:
            return 0
        tokens = self.pool.tokens(rid)
        n = min(len(tokens), max_tokens) // self.pool.block_size * self.pool.block_size
        if n == 0:
            return 0
        key = (namespace, tuple(tokens[:n]))
        if key in self.entries:
            self.entries.move_to_end(key)
            return n
        bids = tuple(self.pool.tables[rid][: n // self.pool.block_size])
        self.pool.retain(bids)
        self.entries[key] = bids
        while len(self.entries) > self.max_entries:
            self.evict_one()
        return n

    def attach(self, rid: str, namespace: str, prompt: tuple[int, ...]) -> int:
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("nonempty cache namespace required")
        if any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("invalid prefix token IDs")
        # Recompute at least the last prompt token to produce logits.
        limit = (len(prompt) - 1) // self.pool.block_size * self.pool.block_size
        matches = []
        for key, bids in self.entries.items():
            if key[0] != namespace:
                continue
            n = min(len(key[1]), limit)
            while n > 0 and prompt[:n] != key[1][:n]:
                n -= self.pool.block_size
            if n:
                matches.append((key, n))
        if not matches:
            return 0
        key, n = max(matches, key=lambda item: item[1])
        bids = self.entries[key][:n // self.pool.block_size]
        self.pool.adopt(rid, bids)
        self.entries.move_to_end(key)
        return n

    def evict_one(self) -> bool:
        if not self.entries:
            return False
        _, bids = self.entries.popitem(last=False)
        self.pool.unpin(bids)
        return True

    def clear(self) -> None:
        while self.evict_one():
            pass
