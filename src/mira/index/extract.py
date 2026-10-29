"""Language-agnostic symbol extractor using heuristic patterns.

Finds functions/classes by name in Python, JS/TS, Go, Rust, Java.
Handles decorators, indentation-based and brace-based scoping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Languages that use indentation-based scoping
_INDENTATION_LANGUAGES = {"python", "py"}

# Languages that use brace-based scoping
_BRACE_LANGUAGES = {
    "javascript",
    "js",
    "typescript",
    "ts",
    "tsx",
    "jsx",
    "go",
    "rust",
    "rs",
    "java",
    "c",
    "cpp",
    "cs",
    "swift",
    "kotlin",
    "kt",
    "scala",
    "php",
}

# Python patterns
_PY_DEF = re.compile(r"^(\s*)(async\s+)?def\s+(\w+)")
_PY_CLASS = re.compile(r"^(\s*)class\s+(\w+)")
_PY_DECORATOR = re.compile(r"^\s*@")

# JS/TS patterns
_JS_FUNCTION = re.compile(r"^(\s*)(?:export\s+(?:default\s+)?)?(?:async\s+)?function\s*\*?\s+(\w+)")
_JS_ARROW = re.compile(
    r"^(\s*)(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::[^=]+)?=\s*(?:async\s+)?"
    r"(?:\([^)]*\)|\w+)\s*(?::\s*[^=]+)?=>"
)
_JS_CLASS = re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?class\s+(\w+)")
_JS_INTERFACE = re.compile(r"^(\s*)(?:export\s+)?(?:default\s+)?interface\s+(\w+)")
_JS_TYPE = re.compile(r"^(\s*)(?:export\s+)?type\s+(\w+)\s*=")
_JS_ENUM = re.compile(r"^(\s*)(?:export\s+)?(?:const\s+)?enum\s+(\w+)")
# Group 3 is "{" when the declaration opens a body on this line.
_JS_METHOD = re.compile(
    r"^(\s*)(?:(?:public|private|protected|static|readonly|async|get|set|abstract|override)\s+)*"
    r"(\w+)\s*(?:<[^>]*>)?\([^)]*\)\s*(?::\s*[^;{=]+\s*;)?\s*(\{)?\s*$"
)

# Go patterns
_GO_FUNC = re.compile(r"^func\s+(?:\(\s*\w+\s+\*?(\w+)(?:\[[^\]]*\])?\s*\)\s+)?(\w+)\s*\(")
_GO_TYPE = re.compile(r"^type\s+(\w+)\s+(struct|interface)\s*\{")

# Rust patterns
_RUST_FN = re.compile(r"^(\s*)(?:pub\s+)?(?:async\s+)?fn\s+(\w+)")
_RUST_STRUCT = re.compile(r"^(\s*)(?:pub\s+)?struct\s+(\w+)")
_RUST_IMPL = re.compile(r"^(\s*)impl(?:<[^>]*>)?\s+(\w+)")

# Java patterns
_JAVA_METHOD = re.compile(
    r"^(\s*)(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"
    r"(?:\w+(?:<[^>]*>)?(?:\[\])*\s+)(\w+)\s*\("
)
_JAVA_CLASS = re.compile(
    r"^(\s*)(?:(?:public|private|protected|static|abstract|final|sealed|non-sealed|strictfp)\s+)*"
    r"(?:class|interface|enum|record)\s+(\w+)"
)
_JAVA_CTOR = re.compile(r"^(\s*)(?:public|private|protected)\s+(\w+)\s*\(")


@dataclass
class SymbolSpan:
    """A located symbol extracted from source code."""

    name: str
    kind: str  # "function", "class", "method", "struct", "impl"
    start_line: int  # 1-based
    end_line: int  # 1-based, inclusive
    source: str
    # Container-scoped name where one exists, e.g. "ClassName.method" or
    # "Receiver.Method". Empty for top-level symbols.
    qualified_name: str = ""


def extract_symbols(source: str, language: str) -> list[SymbolSpan]:
    """Extract all top-level symbols from source code.

    Args:
        source: The source code text.
        language: Language identifier (e.g. "python", "js", "go").

    Returns:
        List of extracted symbols with their source code spans.
    """
    if not source.strip():
        return []

    style = _detect_style(source, language)
    if style == "indentation":
        return _extract_indentation_based(source)
    return _extract_brace_based(source, language)


def find_symbol_by_name(source: str, language: str, name: str) -> SymbolSpan | None:
    """Find a specific symbol by plain or container-qualified name."""
    for sym in extract_symbols(source, language):
        if name in (sym.name, sym.qualified_name):
            return sym
    return None


def _detect_style(source: str, language: str) -> str:
    """Pick indentation- or brace-based extraction; falls back to a
    def/class keyword scan so unknown extensions on Python files don't
    get misclassified as brace-style by dict literals."""
    lang = language.lower().strip()

    if lang in _INDENTATION_LANGUAGES:
        return "indentation"
    if lang in _BRACE_LANGUAGES:
        return "brace"

    # Unknown language — heuristic: check for Python-style def/class keywords
    lines = source.splitlines()
    has_def_class = any(_PY_DEF.match(line) or _PY_CLASS.match(line) for line in lines)
    if has_def_class:
        return "indentation"

    return "brace"


def _extract_indentation_based(source: str) -> list[SymbolSpan]:
    """Extract symbols from indentation-based languages (Python).

    Methods and nested classes carry a qualified name (``Outer.Inner.method``).
    The bare ``name`` stays the identifier used by call graphs.
    """
    lines = source.splitlines()
    symbols: list[SymbolSpan] = []
    # (qualified class name, indent of the class statement)
    class_stack: list[tuple[str, int]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        def_match = _PY_DEF.match(line)
        cls_match = _PY_CLASS.match(line)
        match = def_match or cls_match

        if match:
            indent = len(match.group(1))
            while class_stack and class_stack[-1][1] >= indent:
                class_stack.pop()
            if def_match:
                name = def_match.group(3)
                kind = "method" if class_stack else "function"
            else:
                assert cls_match is not None
                name = cls_match.group(2)
                kind = "class"
            qualified = f"{class_stack[-1][0]}.{name}" if class_stack else ""

            start = i
            while start > 0 and _PY_DECORATOR.match(lines[start - 1]):
                start -= 1

            end = i + 1
            while end < len(lines):
                next_line = lines[end]
                if not next_line.strip():
                    end += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    break
                end += 1

            body = "\n".join(lines[start:end])
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind=kind,
                    start_line=start + 1,
                    end_line=end,
                    source=body,
                    qualified_name=qualified,
                )
            )

            # Stay on the class line so inner methods get found on the next iteration.
            if kind == "class":
                class_stack.append((qualified or name, indent))
                i += 1
            else:
                i = end
        else:
            i += 1

    return symbols


def _extract_brace_based(source: str, language: str) -> list[SymbolSpan]:
    """Extract symbols from brace-based languages."""
    lang = language.lower().strip()
    lines = source.splitlines()

    if lang in ("go",):
        return _extract_go(lines)
    if lang in ("rust", "rs"):
        return _extract_rust(lines)
    if lang in ("java",):
        return _extract_java(lines)
    # Default: JS/TS/C-like
    return _extract_js_ts(lines)


_JS_CONTROL = {
    "if",
    "for",
    "while",
    "switch",
    "catch",
    "else",
    "do",
    "try",
    "return",
    "new",
    "typeof",
    "instanceof",
    "await",
    "yield",
    "void",
    "delete",
    "import",
    "export",
    "function",
    "class",
    "case",
    "var",
    "let",
    "const",
    "super",
    "this",
}


def _extract_js_ts(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from JavaScript/TypeScript.

    Class bodies are descended into so methods are symbols too. A method
    whose line has no opening brace is a single-line declaration; the brace
    matcher is not asked to swallow the rest of the file.
    """
    symbols: list[SymbolSpan] = []
    # (qualified class name, line index of the class's closing brace)
    class_stack: list[tuple[str, int]] = []

    i = 0
    while i < len(lines):
        while class_stack and class_stack[-1][1] < i:
            class_stack.pop()
        line = lines[i]

        cls_match = _JS_CLASS.match(line)
        if cls_match:
            end = _find_brace_end(lines, i)
            name = cls_match.group(2)
            qualified = f"{class_stack[-1][0]}.{name}" if class_stack else ""
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="class",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=qualified,
                )
            )
            class_stack.append((qualified or name, end))
            i += 1
            continue

        declared = _extract_js_declaration(lines, i)
        if declared is not None:
            symbols.append(declared[0])
            i = declared[1]
            continue

        if class_stack:
            method_match = _JS_METHOD.match(line)
            if method_match and method_match.group(2) not in _JS_CONTROL:
                name = method_match.group(2)
                has_brace = method_match.group(3) == "{"
                end = _find_brace_end(lines, i) if has_brace else i
                symbols.append(
                    SymbolSpan(
                        name=name,
                        kind="method",
                        start_line=i + 1,
                        end_line=end + 1,
                        source="\n".join(lines[i : end + 1]),
                        qualified_name=f"{class_stack[-1][0]}.{name}",
                    )
                )
                i = end + 1
                continue

        fn_match = _JS_FUNCTION.match(line)
        if fn_match:
            end = _find_brace_end(lines, i)
            name = fn_match.group(2)
            qualified = f"{class_stack[-1][0]}.{name}" if class_stack else ""
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=qualified,
                )
            )
            i = end + 1
            continue

        arrow_match = _JS_ARROW.match(line)
        if arrow_match:
            end = _find_brace_end(lines, i)
            name = arrow_match.group(2)
            qualified = f"{class_stack[-1][0]}.{name}" if class_stack else ""
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=qualified,
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_js_declaration(lines: list[str], i: int) -> tuple[SymbolSpan, int] | None:
    """Interface, type alias, or enum. Returns the symbol and the next index."""
    line = lines[i]
    iface = _JS_INTERFACE.match(line)
    enum = _JS_ENUM.match(line)
    alias = _JS_TYPE.match(line)
    match = iface or enum or alias
    if match is None:
        return None
    end = _find_brace_end(lines, i) if iface or enum or "{" in line else i
    if iface:
        kind = "interface"
    elif enum:
        kind = "enum"
    else:
        kind = "type"
    name = match.group(2)
    return (
        SymbolSpan(
            name=name,
            kind=kind,
            start_line=i + 1,
            end_line=end + 1,
            source="\n".join(lines[i : end + 1]),
        ),
        end + 1,
    )


def _extract_go(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Go."""
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        fn_match = _GO_FUNC.match(line)
        if fn_match:
            receiver, name = fn_match.group(1), fn_match.group(2)
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="method" if receiver else "function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=f"{receiver}.{name}" if receiver else "",
                )
            )
            i = end + 1
            continue

        type_match = _GO_TYPE.match(line)
        if type_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=type_match.group(1),
                    kind=type_match.group(2),
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_rust(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Rust."""
    symbols: list[SymbolSpan] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        impl_match = _RUST_IMPL.match(line)
        if impl_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=impl_match.group(2),
                    kind="impl",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        struct_match = _RUST_STRUCT.match(line)
        if struct_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=struct_match.group(2),
                    kind="struct",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        fn_match = _RUST_FN.match(line)
        if fn_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=fn_match.group(2),
                    kind="function",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _extract_java(lines: list[str]) -> list[SymbolSpan]:
    """Extract symbols from Java.

    Descends into class/interface/enum/record bodies (every Java method
    lives inside one) and qualifies methods with their enclosing type,
    mirroring the Python extractor. Method bodies are skipped over so
    statements like `return foo(...)` can't false-match.
    """
    symbols: list[SymbolSpan] = []
    enclosing: list[tuple[str, int]] = []  # (type name, body end index)

    i = 0
    while i < len(lines):
        while enclosing and enclosing[-1][1] < i:
            enclosing.pop()
        line = lines[i]

        cls_match = _JAVA_CLASS.match(line)
        if cls_match:
            end = _find_brace_end(lines, i)
            symbols.append(
                SymbolSpan(
                    name=cls_match.group(2),
                    kind="class",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                )
            )
            enclosing.append((cls_match.group(2), end))
            i += 1
            continue

        method_match = _JAVA_METHOD.match(line)
        if not method_match and enclosing:
            ctor = _JAVA_CTOR.match(line)
            if ctor and ctor.group(2) == enclosing[-1][0]:
                method_match = ctor
        if method_match and enclosing:
            # Abstract/interface declarations have no body — span is the line.
            braceless = ";" in line and "{" not in line
            end = i if braceless else _find_brace_end(lines, i)
            name = method_match.group(2)
            symbols.append(
                SymbolSpan(
                    name=name,
                    kind="method",
                    start_line=i + 1,
                    end_line=end + 1,
                    source="\n".join(lines[i : end + 1]),
                    qualified_name=f"{enclosing[-1][0]}.{name}",
                )
            )
            i = end + 1
            continue

        i += 1

    return symbols


def _find_brace_end(lines: list[str], start: int) -> int:
    """Find the line where braces balance out, starting from a given line.

    Scans from ``start`` counting ``{`` and ``}`` until the count returns to zero.
    Returns the line index of the closing brace. If braces never balance,
    returns the last line index.
    """
    depth = 0
    for i in range(start, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
    return len(lines) - 1
