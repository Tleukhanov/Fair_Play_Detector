"""Static checks for the grounded LLM explanation UI.

No browser is required: these tests pin the contract between the API shape and
the no-build frontend — the explanation card, the new pure formatters, the scan
stage label and the methodology section. They also keep the project rule that
API values are rendered as text nodes, never as HTML strings.
"""

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parents[1] / "apps" / "web"
JS_DIR = WEB_DIR / "js"

CHANGED_JS = [
    JS_DIR / "format.js",
    JS_DIR / "views" / "result.js",
    JS_DIR / "views" / "scan.js",
    JS_DIR / "views" / "methodology.js",
]

FORMAT_HELPERS = [
    "explanationSourceLabel",
    "directionLabel",
    "directionClass",
    "formatContribution",
    "formatContributionShare",
    "sourceTitle",
    "fallbackReasonLabel",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_result_view_renders_explanation_fields() -> None:
    source = read(JS_DIR / "views" / "result.js")

    for marker in (
        "verdict?.explanation",
        "explanation.reasoning",
        "explanation.sources",
        "fallback_reason",
        "Почему такой вердикт",
        "Объяснение модели, не обвинение",
        "Источники методики",
        "LLM недоступен — объяснение сформировано детерминированно",
    ):
        assert marker in source

    for css_class in (
        "explanation-card",
        "explanation-head",
        "explanation-subtitle",
        "source-badge",
        "reasoning-list",
        "reasoning-item",
        "reasoning-code",
        "reasoning-body",
        "llm-meta",
        "evidence-contribution",
    ):
        assert css_class in source, f"result.js should use .{css_class}"


def test_format_exports_explanation_helpers() -> None:
    source = read(JS_DIR / "format.js")

    for helper in FORMAT_HELPERS:
        assert f"export function {helper}(" in source, f"missing {helper}"


def test_format_labels_match_api_contract() -> None:
    source = read(JS_DIR / "format.js")

    assert '"LLM"' in source
    assert '"без LLM"' in source
    assert "повышает риск" in source
    assert "снижает риск" in source
    assert "нейтрально" in source
    assert "supports_risk" in source
    assert "reduces_risk" in source
    assert "no_api_key" in source
    assert "validation_failed" in source


def test_scan_stage_label_describes_explanation() -> None:
    source = read(JS_DIR / "views" / "scan.js")

    assert 'explaining: "Формируем объяснение"' in source
    assert "Формируем вердикт" not in source


def test_methodology_explains_llm_limits() -> None:
    source = read(JS_DIR / "views" / "methodology.js")

    assert "Как LLM объясняет вердикт" in source
    assert "не обвиняет" in source
    assert "только факты" in source
    assert "не меняет score" in source
    assert "PGN" in source


def test_explanation_styles_are_present() -> None:
    css = read(WEB_DIR / "styles.css")

    for css_class in (
        ".explanation-card",
        ".explanation-head",
        ".explanation-subtitle",
        ".reasoning-list",
        ".reasoning-item",
        ".reasoning-head",
        ".reasoning-code",
        ".reasoning-body",
        ".direction-up",
        ".direction-down",
        ".direction-neutral",
        ".source-badge",
        ".source-list",
        ".llm-meta",
        ".evidence-contribution",
    ):
        assert css_class in css, f"missing {css_class} in styles.css"
    assert "@media (max-width: 620px)" in css


def test_changed_js_never_injects_html() -> None:
    for path in CHANGED_JS:
        source = read(path).lower()
        for marker in ("innerhtml", "insertadjacenthtml", "outerhtml", "document.write"):
            assert marker not in source, f"{path.name} uses {marker!r}"


def test_explanation_card_is_not_rendered_for_old_verdicts() -> None:
    source = read(JS_DIR / "views" / "result.js")

    # The card builder returns null without an explanation, so old verdicts
    # keep their existing summary/limitations rendering.
    assert "if (explanation === null || typeof explanation !== \"object\") return null;" in source
    assert "].filter(Boolean)" in source


def test_llm_and_fallback_shapes_are_rendered_honestly() -> None:
    source = read(JS_DIR / "views" / "result.js")

    # LLM: blue badge and provider/model metadata, never a fallback alert.
    assert 'const isLlm = source === "llm";' in source
    assert 'badge ${isLlm ? "badge-blue" : "badge-neutral"} source-badge' in source
    assert "explanation?.provider" in source
    assert "explanation?.model" in source

    # Fallback: a reason is shown, but an empty/unknown reason stays silent.
    assert 'if (source !== "deterministic" && reason === "") return null;' in source
    assert "Причина:" in source

    # Deterministic fallback has no sources; hide the section instead of drawing
    # an empty list.
    assert "if (list.length === 0) return null;" in source
    assert "Источники методики" in source
