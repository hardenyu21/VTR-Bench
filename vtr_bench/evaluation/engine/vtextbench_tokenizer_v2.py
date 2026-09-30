#!/usr/bin/env python3
"""Context-aware tokenizer for VTextBench visual-text error scoring.

The tokenizer intentionally ignores ordinary prose punctuation while retaining
symbols that carry content in numbers, measurements, identifiers, and formulae.
It is deterministic, dependency-free, and does not perform case folding or
semantic/spelling normalization.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


TOKENIZER_VERSION = "vtextbench_wer_v2_contextual_20260913"


@dataclass(frozen=True)
class Token:
    value: str
    kind: str


# Ordinary prose punctuation is omitted. These symbols remain because changing
# one of them can change a formula, comparison, price, percentage, or direction.
MATH_OPERATORS = frozenset(
    {
        "=",
        "+",
        "-",
        "−",
        "×",
        "*",
        "÷",
        "→",
        "←",
        "↔",
        "⇒",
        "⇔",
        "<",
        ">",
        "≤",
        "≥",
        "≈",
        "≠",
        "±",
        "∓",
        "^",
        "√",
        "∑",
        "∫",
        "∞",
        "∝",
        "∂",
        "∆",
        "Δ",
        "∇",
        "∈",
        "∉",
        "∩",
        "∪",
        "⊂",
        "⊆",
        "∧",
        "∨",
        "¬",
        "∴",
        "∵",
        "&",
        "~",
        "∮",
        "⁺",
        "⁻",
        "₊",
        "₋",
    }
)
CURRENCY_SYMBOLS = frozenset({"$", "€", "£", "¥", "￥", "₹", "₩", "₽", "¢"})
MATH_BRACKETS = frozenset({"(", ")", "[", "]", "{", "}"})
MATH_SEPARATORS = frozenset({",", ";"})
APOSTROPHES = frozenset({"'", "’", "‘", "ʼ", "＇"})
DASHES = frozenset({"-", "‐", "‑", "–", "—"})
COMPOUND_CONNECTORS = frozenset({"/", "⁄", "∕", "·", "⋅"})
DEGREE_SUFFIXES = frozenset({"C", "F", "K", "R", "N", "S", "E", "W"})
TECHNICAL_UNITS = frozenset(
    {
        "A",
        "Ah",
        "B",
        "Bq",
        "C",
        "cd",
        "cm",
        "d",
        "dB",
        "F",
        "fps",
        "g",
        "GeV",
        "GHz",
        "h",
        "Hz",
        "I",
        "J",
        "K",
        "kg",
        "kHz",
        "kJ",
        "km",
        "kN",
        "kPa",
        "kW",
        "L",
        "lm",
        "lx",
        "m",
        "mA",
        "mg",
        "MHz",
        "min",
        "mJ",
        "mL",
        "ml",
        "mm",
        "mol",
        "MPa",
        "ms",
        "mS",
        "mV",
        "mW",
        "N",
        "nm",
        "Pa",
        "pixel",
        "pixels",
        "PSI",
        "rad",
        "rpm",
        "s",
        "sec",
        "T",
        "V",
        "W",
        "Wh",
        "µg",
        "μg",
        "µm",
        "μm",
        "µmol",
        "μmol",
        "Ω",
        "°C",
        "°F",
        "°K",
    }
)

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>{}\[\]]+", re.IGNORECASE)
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
)
_STRUCTURED_ID_RE = re.compile(
    r"(?:[A-Z]{2,}|[A-Za-z]*\d+[A-Za-z0-9]*)"
    r"(?:\s*[-‐‑–—]\s*(?:[A-Za-z]*\d+[A-Za-z0-9]*|\d+))+"
    r"(?![A-Za-z0-9.])"
)
_SINGLE_LETTER_ID_RE = re.compile(r"[A-Z][-‐‑–—]\d+(?![A-Za-z0-9.])")
_VERSION_RE = re.compile(r"[A-Za-z]*\d+(?:\.\d+){1,}")
_DOTTED_ABBREVIATION_RE = re.compile(r"(?:[A-Za-z]\.){2,}[A-Za-z]?\.?")
_CELL_NOTATION_RE = re.compile(
    r"[A-Z][a-z]?[^|]{0,20}\([^)]{1,8}\)\s*\|[^|]{0,80}\|\|"
)


def normalize_text(text: str) -> str:
    """Apply only representation-level normalization.

    NFC preserves compatibility distinctions such as superscripts and full-width
    glyphs. All Unicode whitespace is collapsed because layout is not part of WER.
    """

    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def is_han(character: str) -> bool:
    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x323AF
    )


def is_kana(character: str) -> bool:
    code = ord(character)
    return (
        0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xFF66 <= code <= 0xFF9D
    )


def _is_word_character(character: str) -> bool:
    category = unicodedata.category(character)
    return category[0] in {"L", "N", "M"} or character == "_"


def _is_digit(character: str) -> bool:
    return unicodedata.category(character) == "Nd"


def _is_variation_selector(character: str) -> bool:
    code = ord(character)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def _is_emoji_modifier(character: str) -> bool:
    return 0x1F3FB <= ord(character) <= 0x1F3FF


def _consume_grapheme(text: str, index: int) -> tuple[str, int]:
    """Consume a practical extended grapheme, including emoji ZWJ sequences."""

    end = index + 1
    while end < len(text):
        character = text[end]
        if unicodedata.category(character).startswith("M"):
            end += 1
            continue
        if _is_variation_selector(character) or _is_emoji_modifier(character):
            end += 1
            continue
        if character == "\u200d" and end + 1 < len(text):
            end += 2
            continue
        break
    return text[index:end], end


def _consume_number(text: str, index: int) -> tuple[str, int] | None:
    start = index
    if text[index] in {"+", "-", "−"}:
        if index + 1 >= len(text) or not (_is_digit(text[index + 1]) or text[index + 1] == "."):
            return None
        index += 1

    if index < len(text) and text[index] == ".":
        if index + 1 >= len(text) or not _is_digit(text[index + 1]):
            return None
        index += 1

    digit_seen = False
    while index < len(text) and _is_digit(text[index]):
        digit_seen = True
        index += 1
    if not digit_seen:
        return None

    # Preserve decimal/thousands punctuation and time/ratio colons inside the
    # numeric token. Repeated dots are accepted for numeric version/IP strings.
    while index + 1 < len(text) and text[index] in {".", ",", ":"} and _is_digit(text[index + 1]):
        index += 1
        while index < len(text) and _is_digit(text[index]):
            index += 1

    if index < len(text) and text[index] in {"e", "E"}:
        exponent = index
        index += 1
        if index < len(text) and text[index] in {"+", "-", "−"}:
            index += 1
        exponent_digits = index
        while index < len(text) and _is_digit(text[index]):
            index += 1
        if exponent_digits == index:
            index = exponent

    return text[start:index], index


def _peek_word_segment(text: str, index: int) -> tuple[str, int] | None:
    if index >= len(text) or not _is_word_character(text[index]) or is_han(text[index]) or is_kana(text[index]):
        return None
    end = index
    pieces: list[str] = []
    while end < len(text):
        character = text[end]
        if _is_word_character(character) and not is_han(character) and not is_kana(character):
            pieces.append(character)
            end += 1
            continue
        # Apostrophes in ordinary words carry little scene-text information.
        # Glue the surrounding letters so don't and dont compare equally.
        if (
            character in APOSTROPHES
            and pieces
            and end + 1 < len(text)
            and _is_word_character(text[end + 1])
            and not is_han(text[end + 1])
            and not is_kana(text[end + 1])
        ):
            end += 1
            continue
        break
    return "".join(pieces), end


def _consume_compound_atom(text: str, index: int) -> tuple[str, int] | None:
    number = _consume_number(text, index) if index < len(text) else None
    if number is not None:
        return number
    if index < len(text) and text[index] == "°":
        end = index + 1
        while end < len(text) and text[end].isspace():
            end += 1
        unit = _peek_word_segment(text, end)
        if unit is not None and unit[0] in DEGREE_SUFFIXES:
            return "°" + unit[0], unit[1]
        return "°", index + 1
    return _peek_word_segment(text, index)


def _match_compound(text: str, index: int) -> tuple[str, int] | None:
    """Match slash/middle-dot expressions and remove connector whitespace.

    Examples: kJ/mol, kJ / mol, A / B, kg · m / s².
    """

    first = _consume_compound_atom(text, index)
    if first is None:
        return None
    atom_values = [first[0]]
    atom_ends = [first[1]]
    connectors: list[str] = []
    connector_spacing: list[bool] = []
    cursor = first[1]
    while True:
        probe = cursor
        while probe < len(text) and text[probe].isspace():
            probe += 1
        space_before = probe > cursor
        if probe >= len(text) or text[probe] not in COMPOUND_CONNECTORS:
            break
        connector = text[probe]
        probe += 1
        after_connector = probe
        while probe < len(text) and text[probe].isspace():
            probe += 1
        space_after = probe > after_connector
        atom = _consume_compound_atom(text, probe)
        if atom is None:
            break
        canonical = "/" if connector in {"/", "⁄", "∕"} else "·"
        connectors.append(canonical)
        connector_spacing.append(space_before or space_after)
        atom_values.append(atom[0])
        atom_ends.append(atom[1])
        cursor = atom[1]
    if not connectors:
        return None
    # A later layout separator must not invalidate an earlier valid unit. Return
    # the longest valid prefix, e.g. µg/m³ from "µg/m³ · +8 Pa".
    for connector_count in range(len(connectors), 0, -1):
        atoms = atom_values[: connector_count + 1]
        used_connectors = connectors[:connector_count]
        used_spacing = connector_spacing[:connector_count]
        if not _compound_sequence_valid(atoms, used_connectors, used_spacing):
            continue
        parts = [atoms[0]]
        for connector, atom in zip(used_connectors, atoms[1:]):
            parts.extend((connector, atom))
        return "".join(parts), atom_ends[connector_count]
    return None


def _compound_sequence_valid(
    atoms: list[str], connectors: list[str], spacing: list[bool]
) -> bool:
    if not all(_is_technical_compound_atom(value) for value in atoms):
        return False
    roles = [_compound_atom_role(value) for value in atoms]
    for position, (connector, spaced) in enumerate(zip(connectors, spacing)):
        left_role, right_role = roles[position], roles[position + 1]
        # A number divided by an unknown two-letter label is usually a field
        # separator (OD -2.50 / OS -2.75), not a unit or formula atom.
        if connector == "/" and {left_role, right_role} == {"number", "technical"}:
            return False
        if connector != "·" or not spaced:
            continue
        # A spaced middle dot is very commonly a layout separator. Preserve it
        # only for unambiguous multiplication among variables, units, or numbers.
        if not (
            set(roles) <= {"variable", "unit"}
            or set(roles) == {"number"}
        ):
            return False
    return True


def _is_technical_compound_atom(value: str) -> bool:
    """Reject prose alternatives while retaining units and formula atoms."""

    core = value.lstrip("+-−")
    if len(core) <= 3:
        return True
    if any(character.isdigit() for character in core):
        return True
    if any(
        "GREEK" in unicodedata.name(character, "")
        or "SUBSCRIPT" in unicodedata.name(character, "")
        or "SUPERSCRIPT" in unicodedata.name(character, "")
        for character in core
    ):
        return True
    return core in {"byte", "bytes", "bit", "bits", "pixel", "pixels", "frame", "frames"}


def _compound_atom_role(value: str) -> str:
    core = value.lstrip("+-−")
    number = _consume_number(core, 0) if core else None
    if number is not None and number[1] == len(core):
        return "number"
    if core in TECHNICAL_UNITS:
        return "unit"
    base_letters = [character for character in core if character.isalpha()]
    if len(base_letters) == 1 and all(
        character.isalpha()
        or unicodedata.category(character).startswith("N")
        or unicodedata.category(character).startswith("M")
        or "SUBSCRIPT" in unicodedata.name(character, "")
        or "SUPERSCRIPT" in unicodedata.name(character, "")
        for character in core
    ):
        return "variable"
    return "technical" if _is_technical_compound_atom(value) else "word"


def _looks_like_math(text: str) -> bool:
    if any(character in MATH_OPERATORS - {"-", "&"} for character in text):
        return True
    if re.search(
        r"(?:^|[^\w])(?:[A-Za-zΑ-ω]|sin|cos|tan|log|ln|exp|max|min)"
        r"\s*[\(\[]\s*[A-Za-zΑ-ω0-9]",
        text,
    ):
        return True
    if re.search(
        r"(?:^|[^\w])(?:[A-Za-zΑ-ω]|\d+)\s*-\s*(?:[A-Za-zΑ-ω]|\d+)(?:$|[^\w])",
        text,
    ):
        return True
    if re.search(r"[/⁄∕][\(\[]", text):
        return True
    if re.search(
        r"[\(\[\{]\s*(?:[+\-−]?\d|[A-Za-zΑ-ω])[^\)\]\}]{0,80}"
        r"[,;][^\)\]\}]{0,80}[\)\]\}]",
        text,
    ):
        return True
    if _looks_like_cell_notation(text):
        return True
    if re.search(r"\|[^\s|](?:[^|]{0,78}[^\s|])?\|", text):
        return True
    return False


def _neighbor_atom(text: str, index: int, step: int) -> str:
    while 0 <= index < len(text) and text[index].isspace():
        index += step
    if not 0 <= index < len(text):
        return ""
    if _is_digit(text[index]):
        return "number"
    if text[index].isalpha():
        # A single-letter variable is math-like; a prose word is not.
        adjacent = index + step
        if not 0 <= adjacent < len(text) or not text[adjacent].isalpha():
            return "variable"
        return "word"
    group_characters = {")", "]", "}"} if step < 0 else {"(", "[", "{"}
    if text[index] in group_characters:
        return "group"
    return ""


def _is_local_minus_operator(text: str, index: int) -> bool:
    left = _neighbor_atom(text, index - 1, -1)
    right = _neighbor_atom(text, index + 1, 1)
    if not left or not right:
        return False
    if "word" in {left, right}:
        return False
    return True


def _is_local_vertical_bar(text: str, index: int) -> bool:
    left = _neighbor_atom(text, index - 1, -1)
    right = _neighbor_atom(text, index + 1, 1)
    return bool(left and right and "word" not in {left, right})


def _is_paired_vertical_bar(text: str, index: int) -> bool:
    if index + 1 < len(text) and not text[index + 1].isspace():
        closing = text.find("|", index + 1)
        if closing > index + 1 and not text[closing - 1].isspace():
            return True
    if index > 0 and not text[index - 1].isspace():
        opening = text.rfind("|", 0, index)
        if opening >= 0 and opening + 1 < index and not text[opening + 1].isspace():
            return True
    return False


def _looks_like_cell_notation(text: str) -> bool:
    return bool(_CELL_NOTATION_RE.search(text))


def _inside_math_group(text: str, index: int) -> bool:
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}")):
        if text.rfind(opening, 0, index) > text.rfind(closing, 0, index) and text.find(
            closing, index + 1
        ) >= 0:
            return True
    return False


def _is_local_middle_dot_operator(text: str, index: int) -> bool:
    left = _neighbor_atom(text, index - 1, -1)
    right = _neighbor_atom(text, index + 1, 1)
    return left == "group" and right == "variable"


def _is_clause_boundary(text: str, index: int) -> bool:
    character = text[index]
    if character in {";", ":", "!", "?"}:
        return True
    if character != ".":
        return False
    left_digit = index > 0 and _is_digit(text[index - 1])
    right_digit = index + 1 < len(text) and _is_digit(text[index + 1])
    return not (left_digit and right_digit)


def _is_local_formula_slash(text: str, index: int) -> bool:
    left = index - 1
    while left >= 0 and text[left].isspace():
        left -= 1
    right = index + 1
    while right < len(text) and text[right].isspace():
        right += 1
    if left >= 0 and text[left] in {")", "]", "}"}:
        return True
    if right == index + 1 and right < len(text) and text[right] in {"(", "[", "{"}:
        return True

    start = index - 1
    while start >= 0 and not _is_clause_boundary(text, start):
        start -= 1
    end = index + 1
    while end < len(text) and not _is_clause_boundary(text, end):
        end += 1
    clause = text[start + 1 : end]
    return any(
        character in MATH_OPERATORS - {"-", "&"}
        for character in clause
    )


def _strip_url_tail(value: str) -> tuple[str, str]:
    # Sentence-final punctuation is not part of a URL. URL-internal punctuation
    # remains semantic and is preserved.
    tail = ""
    while value and value[-1] in {".", ",", ";", ":", "!", "?"}:
        tail = value[-1] + tail
        value = value[:-1]
    return value, tail


def tokenize(text: str) -> list[Token]:
    normalized = normalize_text(text)
    math_context = _looks_like_math(normalized)
    tokens: list[Token] = []
    index = 0
    bracket_depth = 0

    while index < len(normalized):
        character = normalized[index]
        if character.isspace():
            index += 1
            continue

        if normalized.startswith("<UNK>", index):
            tokens.append(Token("<UNK>", "unknown"))
            index += 5
            continue

        url = _URL_RE.match(normalized, index)
        if url:
            value, _ = _strip_url_tail(url.group())
            if value:
                tokens.append(Token(value, "url"))
            index = url.end()
            continue

        email = _EMAIL_RE.match(normalized, index)
        if email:
            tokens.append(Token(email.group(), "email"))
            index = email.end()
            continue

        if character in {"#", "@"} and index + 1 < len(normalized):
            segment = _peek_word_segment(normalized, index + 1)
            if segment:
                tokens.append(Token(character + segment[0], "social_identifier"))
                index = segment[1]
                continue

        structured_id = _STRUCTURED_ID_RE.match(normalized, index)
        if structured_id:
            value = re.sub(r"\s*[-‐‑–—]\s*", "-", structured_id.group())
            tokens.append(Token(value, "identifier"))
            index = structured_id.end()
            continue

        single_letter_id = _SINGLE_LETTER_ID_RE.match(normalized, index)
        if single_letter_id:
            value = re.sub(r"[-‐‑–—]", "-", single_letter_id.group())
            tokens.append(Token(value, "identifier"))
            index = single_letter_id.end()
            continue

        version = _VERSION_RE.match(normalized, index)
        if version:
            tokens.append(Token(version.group(), "identifier"))
            index = version.end()
            continue

        abbreviation = _DOTTED_ABBREVIATION_RE.match(normalized, index)
        if abbreviation:
            value = abbreviation.group().replace(".", "")
            tokens.append(Token(value, "word"))
            index = abbreviation.end()
            continue

        compound = _match_compound(normalized, index)
        if compound:
            tokens.append(Token(compound[0], "compound"))
            index = compound[1]
            continue

        number = _consume_number(normalized, index)
        if number:
            tokens.append(Token(number[0], "number"))
            index = number[1]
            continue

        if character == "°":
            unit = _consume_compound_atom(normalized, index)
            assert unit is not None
            tokens.append(Token(unit[0], "unit"))
            index = unit[1]
            continue

        if is_han(character):
            grapheme, index = _consume_grapheme(normalized, index)
            tokens.append(Token(grapheme, "han"))
            continue

        if is_kana(character):
            grapheme, index = _consume_grapheme(normalized, index)
            tokens.append(Token(grapheme, "kana"))
            continue

        word = _peek_word_segment(normalized, index)
        if word:
            tokens.append(Token(word[0], "word"))
            index = word[1]
            continue

        if character in CURRENCY_SYMBOLS:
            tokens.append(Token(character, "currency"))
            index += 1
            continue

        if character in {"%", "％"}:
            tokens.append(Token("%", "unit"))
            index += 1
            continue

        if character in MATH_OPERATORS and (
            character != "-" or _is_local_minus_operator(normalized, index)
        ):
            tokens.append(Token(character, "operator"))
            index += 1
            continue

        if character == "|" and (
            (
                math_context
                and _inside_math_group(normalized, index)
                and _is_local_vertical_bar(normalized, index)
            )
            or (
                math_context
                and (
                    _looks_like_cell_notation(normalized)
                    or _is_paired_vertical_bar(normalized, index)
                )
            )
        ):
            tokens.append(Token(character, "operator"))
            index += 1
            continue

        if character in {"/", "⁄", "∕"} and _is_local_formula_slash(
            normalized, index
        ):
            tokens.append(Token("/", "operator"))
            index += 1
            continue

        if character in {"·", "⋅"} and _is_local_middle_dot_operator(normalized, index):
            tokens.append(Token("·", "operator"))
            index += 1
            continue

        if character in MATH_BRACKETS and math_context:
            tokens.append(Token(character, "bracket"))
            if character in {"(", "[", "{"}:
                bracket_depth += 1
            else:
                bracket_depth = max(0, bracket_depth - 1)
            index += 1
            continue

        if character in MATH_SEPARATORS and math_context and bracket_depth:
            tokens.append(Token(character, "formula_separator"))
            index += 1
            continue

        category = unicodedata.category(character)
        if category == "So":
            grapheme, index = _consume_grapheme(normalized, index)
            tokens.append(Token(grapheme, "symbol"))
            continue

        # All remaining punctuation is ordinary prose punctuation and is
        # deliberately ignored. This includes sentence-final punctuation,
        # quotes, colons, semicolons, and prose dashes.
        index += 1

    return tokens


def wer_tokens(text: str) -> list[str]:
    return [token.value for token in tokenize(text)]


def tokenization_record(text: str) -> dict[str, object]:
    detailed = tokenize(text)
    return {
        "tokenizer_version": TOKENIZER_VERSION,
        "input": text,
        "normalized": normalize_text(text),
        "tokens": [token.value for token in detailed],
        "token_kinds": [token.kind for token in detailed],
    }


__all__ = [
    "TOKENIZER_VERSION",
    "Token",
    "normalize_text",
    "tokenize",
    "tokenization_record",
    "wer_tokens",
]
