"""Interpret a URL; this module sends no network request."""
from urllib.parse import urlsplit, parse_qs

def describe(url: str) -> dict:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("http(s) URL required")
    if parts.username or parts.password:
        raise ValueError("credentials in URL are not supported")
    return {"scheme": parts.scheme, "host": parts.hostname,
            "port": parts.port or (443 if parts.scheme == "https" else 80),
            "path": parts.path or "/", "query": parse_qs(parts.query),
            "fragment_sent_to_server": False}

if __name__ == "__main__":
    print(describe("http://127.0.0.1:8765/tasks?status=todo&limit=2#local-note"))
