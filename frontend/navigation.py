"""Which pages a role gets. Pure data (no Streamlit calls) so it is easy to test.

A viewer never gets a page with write controls; the API refuses their writes anyway."""

from __future__ import annotations

# (module, title, icon, url_path, admin_only)
SPECS = [
    ("fleet", "Fleet health", "📊", "fleet", False),
    ("investigate", "Investigations", "🔎", "investigations", False),
    ("upload", "Upload data", "⬆️", "upload", True),
    ("jobs", "Jobs", "⚙️", "jobs", False),
    ("admin", "Admin", "🛠️", "admin", True),
]


def specs_for(role: str | None) -> list[tuple[str, str, str, str, bool]]:
    return [s for s in SPECS if role == "admin" or not s[4]]


def titles_for(role: str | None) -> list[str]:
    return [s[1] for s in specs_for(role)]
