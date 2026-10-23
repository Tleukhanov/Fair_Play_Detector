"""Static checks for the security-sensitive parts of the no-build frontend.

These tests do not try to be a browser. They pin the conventions that keep API
strings out of HTML parsing: DOM builders instead of ``innerHTML``, same-origin
assets, and ``rel="noopener noreferrer"`` on new-tab links.
"""

import re
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
JS_FILES = sorted(WEB_DIR.rglob("*.js"))
HTML_FILES = sorted(WEB_DIR.rglob("*.html"))

# API values must be inserted with textContent/DOM nodes, never parsed as HTML.
FORBIDDEN_JS = [
    "innerhtml",
    "insertadjacenthtml",
    "outerhtml",
    "document.write",
    "document.writeln",
    "eval(",
    "new function(",
]


def test_web_files_are_discovered() -> None:
    assert HTML_FILES, "index.html should exist"
    assert len(JS_FILES) >= 10, "all frontend modules should be checked"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda path: path.name)
def test_js_avoids_html_string_injection(path: Path) -> None:
    source = path.read_text(encoding="utf-8").lower()

    for marker in FORBIDDEN_JS:
        assert marker not in source, f"{path.name} uses {marker!r}"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda path: path.name)
def test_js_imports_are_relative_or_bare(path: Path) -> None:
    source = path.read_text(encoding="utf-8")

    for match in re.finditer(r"""from\s+["']([^"']+)["']""", source):
        target = match.group(1)
        assert not target.startswith(("http://", "https://", "//")), (
            f"{path.name} imports a remote module: {target}"
        )


@pytest.mark.parametrize("path", HTML_FILES, ids=lambda path: path.name)
def test_html_has_no_remote_scripts_or_styles(path: Path) -> None:
    html = path.read_text(encoding="utf-8")

    assert not re.search(r"<script[^>]+src=[\"']https?://", html, re.IGNORECASE)
    assert not re.search(r"<link[^>]+href=[\"']https?://", html, re.IGNORECASE)


def test_index_html_new_tab_links_are_hardened() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    anchors = re.findall(r"<a\b[^>]*target=\"_blank\"[^>]*>", html, re.IGNORECASE)
    assert anchors, "the shell should link to GitHub/Lichess"
    for anchor in anchors:
        assert 'rel="noopener noreferrer"' in anchor


def test_dom_builder_hardens_new_tab_links() -> None:
    components = (WEB_DIR / "js" / "components.js").read_text(encoding="utf-8")

    # The shared builder adds rel even when a caller forgets it.
    assert 'node.getAttribute("target") === "_blank"' in components
    assert 'rel.add("noopener")' in components
    assert 'rel.add("noreferrer")' in components


def test_no_remote_assets_or_fonts_in_stylesheet() -> None:
    css = (WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert not re.search(r"@import\s+url\(\s*[\"']?https?://", css, re.IGNORECASE)
    assert "url(http" not in css.lower()
    assert "fonts.googleapis" not in css.lower()
