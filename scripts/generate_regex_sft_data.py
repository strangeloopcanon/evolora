#!/usr/bin/env python3
"""Generate SFT training data for teaching regex capabilities.

Produces a JSONL file with thousands of diverse regex examples covering:
- Basic literals and escaping
- Character classes and predefined classes
- Anchors and boundaries
- Quantifiers (greedy and lazy)
- Alternation and grouping
- Backreferences
- Lookahead and lookbehind
- Common real-world patterns
- Error correction and explanation tasks
"""

from __future__ import annotations

import json
import random
from pathlib import Path


def generate_literal_matching() -> list[dict]:
    """Basic literal string matching."""
    examples = []

    words = [
        "hello",
        "world",
        "python",
        "regex",
        "pattern",
        "match",
        "search",
        "find",
        "replace",
        "test",
        "data",
        "string",
        "text",
        "code",
        "file",
        "user",
        "name",
        "email",
        "phone",
        "date",
        "time",
        "number",
        "value",
        "error",
        "warning",
        "success",
        "failed",
        "start",
        "end",
        "begin",
    ]

    for word in words:
        examples.append(
            {"prompt": f"Write a regex that matches the exact word '{word}'", "completion": word}
        )
        examples.append(
            {
                "prompt": f"What regex pattern matches the literal string '{word}'?",
                "completion": word,
            }
        )

    # Multi-word phrases
    phrases = [
        "hello world",
        "log file",
        "error code",
        "user name",
        "test case",
        "data set",
        "file path",
        "time stamp",
        "end of line",
        "start of file",
    ]
    for phrase in phrases:
        escaped = phrase.replace(" ", r"\s+")
        examples.append(
            {
                "prompt": f"Write a regex to match '{phrase}' with flexible whitespace",
                "completion": escaped,
            }
        )

    return examples


def generate_escaping() -> list[dict]:
    """Special character escaping."""
    examples = []

    special_chars = [
        (".", r"\.", "a period/dot"),
        ("*", r"\*", "an asterisk"),
        ("+", r"\+", "a plus sign"),
        ("?", r"\?", "a question mark"),
        ("^", r"\^", "a caret"),
        ("$", r"\$", "a dollar sign"),
        ("|", r"\|", "a pipe/vertical bar"),
        ("(", r"\(", "an opening parenthesis"),
        (")", r"\)", "a closing parenthesis"),
        ("[", r"\[", "an opening bracket"),
        ("]", r"\]", "a closing bracket"),
        ("{", r"\{", "an opening brace"),
        ("}", r"\}", "a closing brace"),
        ("\\", r"\\", "a backslash"),
    ]

    for char, escaped, desc in special_chars:
        examples.append(
            {"prompt": f"Write a regex that matches {desc} literally", "completion": escaped}
        )
        examples.append(
            {
                "prompt": f"How do I match the character '{char}' literally in a regex?",
                "completion": f"Use {escaped} to match a literal '{char}'",
            }
        )

    # Common patterns requiring escaping
    patterns = [
        ("$100", r"\$100", "the price $100"),
        ("file.txt", r"file\.txt", "the filename file.txt"),
        ("C++", r"C\+\+", "the programming language C++"),
        ("(test)", r"\(test\)", "the string (test) with parentheses"),
        ("a*b", r"a\*b", "the expression a*b"),
        ("x^2", r"x\^2", "x squared (x^2)"),
        ("a|b", r"a\|b", "the string a|b with a pipe"),
        ("path/to/file", r"path/to/file", "a Unix file path"),
        ("C:\\Users", r"C:\\Users", "a Windows path prefix"),
        ("[URGENT]", r"\[URGENT\]", "the tag [URGENT]"),
    ]

    for _literal, escaped, desc in patterns:
        examples.append({"prompt": f"Write a regex to match {desc}", "completion": escaped})

    return examples


def generate_character_classes() -> list[dict]:
    """Character classes [abc], [a-z], [^abc]."""
    examples = []

    # Basic character classes
    classes = [
        ("[aeiou]", "any vowel"),
        ("[bcdfghjklmnpqrstvwxyz]", "any consonant"),
        ("[0-9]", "any digit"),
        ("[a-z]", "any lowercase letter"),
        ("[A-Z]", "any uppercase letter"),
        ("[a-zA-Z]", "any letter (upper or lower case)"),
        ("[a-zA-Z0-9]", "any alphanumeric character"),
        ("[0-9a-fA-F]", "any hexadecimal digit"),
        ("[01]", "a binary digit (0 or 1)"),
        ("[0-7]", "an octal digit"),
        ("[a-z_]", "a lowercase letter or underscore"),
        ("[a-zA-Z_]", "a letter or underscore"),
        (
            "[a-zA-Z_][a-zA-Z0-9_]*",
            "a valid identifier (letter/underscore followed by alphanumerics)",
        ),
    ]

    for pattern, desc in classes:
        examples.append(
            {"prompt": f"Write a regex character class that matches {desc}", "completion": pattern}
        )
        examples.append(
            {"prompt": f"What does the regex {pattern} match?", "completion": f"It matches {desc}"}
        )

    # Negated character classes
    negated = [
        ("[^0-9]", "any character that is NOT a digit"),
        ("[^a-z]", "any character that is NOT a lowercase letter"),
        ("[^aeiou]", "any character that is NOT a vowel"),
        ("[^\\s]", "any non-whitespace character"),
        ("[^a-zA-Z]", "any non-letter character"),
        ("[^0-9a-zA-Z]", "any non-alphanumeric character (punctuation, symbols, etc.)"),
    ]

    for pattern, desc in negated:
        examples.append({"prompt": f"Write a regex that matches {desc}", "completion": pattern})
        examples.append(
            {
                "prompt": f"What does the negated character class {pattern} match?",
                "completion": f"It matches {desc}",
            }
        )

    # Ranges and combinations
    examples.append(
        {
            "prompt": "Write a regex to match a letter from a to f or a digit",
            "completion": "[a-f0-9]",
        }
    )
    examples.append(
        {
            "prompt": "How do I match any character except newline in a character class?",
            "completion": "Use [^\\n] to match any character except newline, or use the dot . metacharacter",
        }
    )

    return examples


def generate_predefined_classes() -> list[dict]:
    """Predefined character classes: \\d, \\w, \\s, etc."""
    examples = []

    predefined = [
        (r"\d", "any digit (0-9)", "[0-9]"),
        (r"\D", "any non-digit", "[^0-9]"),
        (r"\w", "any word character (letter, digit, underscore)", "[a-zA-Z0-9_]"),
        (r"\W", "any non-word character", "[^a-zA-Z0-9_]"),
        (r"\s", "any whitespace (space, tab, newline)", "[ \\t\\n\\r\\f\\v]"),
        (r"\S", "any non-whitespace character", "[^ \\t\\n\\r\\f\\v]"),
    ]

    for shorthand, desc, equivalent in predefined:
        examples.append(
            {
                "prompt": f"What does {shorthand} match in a regex?",
                "completion": f"{shorthand} matches {desc}, equivalent to {equivalent}",
            }
        )
        examples.append(
            {"prompt": f"Write a shorthand regex class for {desc}", "completion": shorthand}
        )
        examples.append(
            {"prompt": f"What is the shorthand for {equivalent} in regex?", "completion": shorthand}
        )

    # Combinations
    combos = [
        (r"\d+", "one or more digits"),
        (r"\w+", "one or more word characters (a word)"),
        (r"\s+", "one or more whitespace characters"),
        (r"\d{3}", "exactly three digits"),
        (r"\w{2,5}", "a word between 2 and 5 characters"),
        (r"\d{1,3}", "one to three digits"),
        (r"\s*", "zero or more whitespace characters"),
        (r"\d+\.\d+", "a decimal number like 3.14"),
        (r"\w+@\w+\.\w+", "a simple email pattern"),
    ]

    for pattern, desc in combos:
        examples.append({"prompt": f"Write a regex to match {desc}", "completion": pattern})

    return examples


def generate_anchors() -> list[dict]:
    """Anchors: ^, $, \\b, \\B."""
    examples = []

    # Line anchors
    examples.extend(
        [
            {
                "prompt": "Write a regex that matches 'hello' only at the start of a line",
                "completion": "^hello",
            },
            {
                "prompt": "Write a regex that matches 'end' only at the end of a line",
                "completion": "end$",
            },
            {
                "prompt": "Write a regex that matches a line containing only 'test'",
                "completion": "^test$",
            },
            {
                "prompt": "What does ^ mean in a regex (outside a character class)?",
                "completion": "^ anchors the match to the start of a line or string",
            },
            {
                "prompt": "What does $ mean in a regex?",
                "completion": "$ anchors the match to the end of a line or string",
            },
            {"prompt": "Write a regex to match empty lines", "completion": "^$"},
            {
                "prompt": "Write a regex to match lines that start with a number",
                "completion": r"^\d",
            },
            {"prompt": "Write a regex to match lines that end with a period", "completion": r"\.$"},
        ]
    )

    # Word boundaries
    examples.extend(
        [
            {
                "prompt": "Write a regex that matches the word 'cat' but not 'cats' or 'concatenate'",
                "completion": r"\bcat\b",
            },
            {
                "prompt": "What does \\b mean in a regex?",
                "completion": r"\b is a word boundary anchor - it matches the position between a word character and a non-word character",
            },
            {"prompt": "Write a regex to match 'log' as a complete word", "completion": r"\blog\b"},
            {
                "prompt": "Write a regex to match words that start with 'pre'",
                "completion": r"\bpre\w*",
            },
            {
                "prompt": "Write a regex to match words that end with 'ing'",
                "completion": r"\w+ing\b",
            },
            {
                "prompt": "Write a regex to match words that end with 'tion'",
                "completion": r"\w+tion\b",
            },
            {
                "prompt": "What does \\B mean in a regex?",
                "completion": r"\B matches a position that is NOT a word boundary - the opposite of \b",
            },
            {
                "prompt": "Write a regex to match 'cat' only when it's part of a larger word",
                "completion": r"\Bcat\B",
            },
        ]
    )

    return examples


def generate_quantifiers() -> list[dict]:
    """Quantifiers: *, +, ?, {n}, {n,}, {n,m}."""
    examples = []

    # Basic quantifiers
    quantifiers = [
        ("*", "zero or more", r"a*", "zero or more 'a' characters"),
        ("+", "one or more", r"a+", "one or more 'a' characters"),
        ("?", "zero or one (optional)", r"a?", "zero or one 'a' character"),
        ("{3}", "exactly n", r"a{3}", "exactly three 'a' characters"),
        ("{2,4}", "between n and m", r"a{2,4}", "between 2 and 4 'a' characters"),
        ("{2,}", "n or more", r"a{2,}", "two or more 'a' characters"),
    ]

    for symbol, meaning, example, desc in quantifiers:
        examples.append(
            {
                "prompt": f"What does the quantifier {symbol} mean in regex?",
                "completion": f"{symbol} means {meaning}. For example, {example} matches {desc}",
            }
        )

    # Practical examples
    practical = [
        (r"colou?r", "both 'color' and 'colour'"),
        (r"https?://", "both 'http://' and 'https://'"),
        (r"\d{3}-\d{4}", "a phone number like 555-1234"),
        (r"\d{3}-\d{3}-\d{4}", "a phone number like 555-123-4567"),
        (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "an IP address pattern"),
        (r"\w+", "one or more word characters (a word)"),
        (r".*", "any character zero or more times (greedy)"),
        (r".+", "any character one or more times"),
        (r"\d{4}-\d{2}-\d{2}", "a date in YYYY-MM-DD format"),
        (r"\d{2}/\d{2}/\d{4}", "a date in MM/DD/YYYY format"),
        (r"[A-Z]{2,3}", "a 2 or 3 letter uppercase code"),
        (r"\d{5}(-\d{4})?", "a US ZIP code (5 digits, optionally followed by -4 digits)"),
        (r"\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{4}", "a phone number with optional country code"),
    ]

    for pattern, desc in practical:
        examples.append({"prompt": f"Write a regex to match {desc}", "completion": pattern})

    return examples


def generate_greedy_vs_lazy() -> list[dict]:
    """Greedy vs lazy quantifiers."""
    examples = []

    examples.extend(
        [
            {
                "prompt": "What is the difference between greedy and lazy quantifiers?",
                "completion": "Greedy quantifiers (*, +, ?) match as much as possible, while lazy quantifiers (*?, +?, ??) match as little as possible",
            },
            {
                "prompt": "How do you make a quantifier lazy in regex?",
                "completion": "Add a ? after the quantifier: *? for lazy zero-or-more, +? for lazy one-or-more, ?? for lazy optional",
            },
            {
                "prompt": "Write a regex to match the shortest string between < and >",
                "completion": "<.+?>",
            },
            {"prompt": "Write a regex to match HTML tags non-greedily", "completion": "<[^>]+?>"},
            {
                "prompt": "What does .*? match?",
                "completion": ".*? is a lazy match for any characters - it matches as few characters as possible",
            },
            {
                "prompt": "Write a regex to extract the first quoted string",
                "completion": '"[^"]*?"' + " or " + '".+?"',
            },
            {
                "prompt": "Why would <.*> fail to match a single HTML tag in '<b>text</b>'?",
                "completion": "Because .* is greedy, it matches '<b>text</b>' (everything from first < to last >). Use <.*?> or <[^>]*> instead",
            },
            {"prompt": "Write a lazy version of a+", "completion": "a+?"},
            {"prompt": "Write a lazy version of \\d{2,5}", "completion": r"\d{2,5}?"},
        ]
    )

    return examples


def generate_alternation() -> list[dict]:
    """Alternation with |."""
    examples = []

    examples.extend(
        [
            {"prompt": "Write a regex that matches either 'cat' or 'dog'", "completion": "cat|dog"},
            {
                "prompt": "Write a regex that matches 'yes', 'no', or 'maybe'",
                "completion": "yes|no|maybe",
            },
            {
                "prompt": "What does | mean in regex?",
                "completion": "| is the alternation operator - it matches either the pattern on the left or the right",
            },
            {
                "prompt": "Write a regex to match common image extensions",
                "completion": r"\.(jpg|jpeg|png|gif|bmp|webp)",
            },
            {
                "prompt": "Write a regex to match 'Mr', 'Mrs', 'Ms', or 'Dr'",
                "completion": "Mr|Mrs|Ms|Dr",
            },
            {
                "prompt": "Write a regex to match days of the week",
                "completion": "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday",
            },
            {
                "prompt": "Write a regex to match common programming languages",
                "completion": "Python|JavaScript|Java|C\\+\\+|Ruby|Go|Rust",
            },
            {
                "prompt": "Write a regex to match HTTP methods",
                "completion": "GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS",
            },
            {
                "prompt": "Write a regex to match true or false (case-insensitive hint)",
                "completion": "[Tt]rue|[Ff]alse",
            },
        ]
    )

    return examples


def generate_grouping() -> list[dict]:
    """Grouping with (), (?:...)."""
    examples = []

    examples.extend(
        [
            {
                "prompt": "What do parentheses () do in a regex?",
                "completion": "Parentheses create a capturing group - they group patterns together and capture the matched text for later use",
            },
            {
                "prompt": "Write a regex with a group to match 'ab' repeated 3 times",
                "completion": "(ab){3}",
            },
            {
                "prompt": "Write a regex to match 'la' repeated one or more times (like 'la', 'lala', 'lalala')",
                "completion": "(la)+",
            },
            {
                "prompt": "What is a non-capturing group and how do you write one?",
                "completion": "A non-capturing group (?:...) groups patterns without capturing. Use (?:pattern) instead of (pattern)",
            },
            {
                "prompt": "Write a regex with a non-capturing group to match 'http' or 'https'",
                "completion": "https?://" + " or " + "(?:https?)://",
            },
            {
                "prompt": "Write a regex to capture the protocol and domain from a URL",
                "completion": r"(https?)://([^/]+)",
            },
            {
                "prompt": "Write a regex to match and capture a name in 'Hello, NAME!' format",
                "completion": r"Hello, (\w+)!",
            },
            {
                "prompt": "Write a regex with groups to parse 'key=value' pairs",
                "completion": r"(\w+)=(\w+)",
            },
            {
                "prompt": "Write a regex to capture area code and number from '(555) 123-4567'",
                "completion": r"\((\d{3})\)\s*(\d{3}-\d{4})",
            },
            {
                "prompt": "Write a regex with groups to capture year, month, day from YYYY-MM-DD",
                "completion": r"(\d{4})-(\d{2})-(\d{2})",
            },
        ]
    )

    return examples


def generate_backreferences() -> list[dict]:
    """Backreferences: \\1, \\2, etc."""
    examples = []

    examples.extend(
        [
            {
                "prompt": "What is a backreference in regex?",
                "completion": r"A backreference like \1 or \2 matches the same text that was matched by a previous capturing group",
            },
            {
                "prompt": "Write a regex to match repeated words like 'the the'",
                "completion": r"\b(\w+)\s+\1\b",
            },
            {
                "prompt": "Write a regex to match strings enclosed in matching quotes (single or double)",
                "completion": r"(['\"]).*?\1",
            },
            {
                "prompt": "Write a regex to match HTML tags with matching open and close tags like <b>...</b>",
                "completion": r"<(\w+)>.*?</\1>",
            },
            {
                "prompt": "How do you reference the first capturing group?",
                "completion": r"Use \1 to reference the first capturing group",
            },
            {
                "prompt": "Write a regex to match lines where the first and last word are the same",
                "completion": r"^(\w+)\b.*\b\1$",
            },
            {
                "prompt": "Write a regex to find doubled characters like 'aa', 'bb', etc.",
                "completion": r"(.)\1",
            },
            {
                "prompt": "Write a regex to match strings like 'abab' or 'cdcd' (two-char pattern repeated)",
                "completion": r"(..)\\1",
            },
        ]
    )

    return examples


def generate_lookahead() -> list[dict]:
    """Lookahead: (?=...), (?!...)."""
    examples = []

    examples.extend(
        [
            {
                "prompt": "What is a positive lookahead in regex?",
                "completion": "(?=pattern) is a positive lookahead - it asserts that what follows matches pattern, without consuming characters",
            },
            {
                "prompt": "What is a negative lookahead in regex?",
                "completion": "(?!pattern) is a negative lookahead - it asserts that what follows does NOT match pattern",
            },
            {
                "prompt": "Write a regex to match 'q' only if followed by 'u'",
                "completion": "q(?=u)",
            },
            {
                "prompt": "Write a regex to match 'q' only if NOT followed by 'u'",
                "completion": "q(?!u)",
            },
            {
                "prompt": "Write a regex to match a number only if followed by 'px'",
                "completion": r"\d+(?=px)",
            },
            {
                "prompt": "Write a regex to match words not followed by a comma",
                "completion": r"\w+(?!,)",
            },
            {
                "prompt": "Write a regex to match 'foo' not followed by 'bar'",
                "completion": "foo(?!bar)",
            },
            {
                "prompt": "Write a regex to validate a password has at least one digit",
                "completion": "(?=.*\\d).+",
            },
            {
                "prompt": "Write a regex to validate a password has at least one uppercase letter",
                "completion": "(?=.*[A-Z]).+",
            },
            {
                "prompt": "Write a regex for a password with at least one digit and one uppercase",
                "completion": "(?=.*\\d)(?=.*[A-Z]).+",
            },
            {
                "prompt": "Write a regex to match .js files but not .json files",
                "completion": r"\.js(?!on)$",
            },
        ]
    )

    return examples


def generate_lookbehind() -> list[dict]:
    """Lookbehind: (?<=...), (?<!...)."""
    examples = []

    examples.extend(
        [
            {
                "prompt": "What is a positive lookbehind in regex?",
                "completion": "(?<=pattern) is a positive lookbehind - it asserts that what precedes matches pattern, without consuming characters",
            },
            {
                "prompt": "What is a negative lookbehind in regex?",
                "completion": "(?<!pattern) is a negative lookbehind - it asserts that what precedes does NOT match pattern",
            },
            {
                "prompt": "Write a regex to match a number only if preceded by '$'",
                "completion": r"(?<=\$)\d+",
            },
            {
                "prompt": "Write a regex to match a number NOT preceded by '$'",
                "completion": r"(?<!\$)\d+",
            },
            {
                "prompt": "Write a regex to match text after 'name: '",
                "completion": r"(?<=name: )\w+",
            },
            {"prompt": "Write a regex to match words after 'Dr. '", "completion": r"(?<=Dr\. )\w+"},
            {
                "prompt": "Write a regex to match file extensions (part after the dot)",
                "completion": r"(?<=\.)\w+$",
            },
            {
                "prompt": "Write a regex to match the domain part of an email (after @)",
                "completion": r"(?<=@)[a-zA-Z0-9.-]+",
            },
        ]
    )

    return examples


def generate_common_patterns() -> list[dict]:
    """Common real-world regex patterns."""
    examples = []

    patterns = [
        # Email
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "an email address"),
        (
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "an email address with word boundaries",
        ),
        # URLs
        (r"https?://[^\s]+", "a URL starting with http or https"),
        (
            r"https?://(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?",
            "a more complete URL pattern",
        ),
        # IP addresses
        (r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "an IPv4 address (basic)"),
        (
            r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
            "a valid IPv4 address",
        ),
        # Phone numbers
        (r"\d{3}-\d{3}-\d{4}", "a US phone number like 555-123-4567"),
        (r"\(\d{3}\)\s*\d{3}-\d{4}", "a phone number like (555) 123-4567"),
        (r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "various US phone number formats"),
        # Dates
        (r"\d{4}-\d{2}-\d{2}", "a date in YYYY-MM-DD format"),
        (r"\d{2}/\d{2}/\d{4}", "a date in MM/DD/YYYY format"),
        (r"\d{1,2}/\d{1,2}/\d{2,4}", "flexible date format like 1/5/2024 or 01/05/24"),
        (
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
            "a date like 'Jan 5, 2024'",
        ),
        # Times
        (r"\d{1,2}:\d{2}", "a time like 9:30 or 14:45"),
        (r"\d{1,2}:\d{2}:\d{2}", "a time with seconds like 9:30:00"),
        (r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)", "a 12-hour time with AM/PM"),
        # Credit cards (basic pattern, not validation)
        (r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}", "a credit card number"),
        # Social Security Number
        (r"\d{3}-\d{2}-\d{4}", "a US Social Security Number format"),
        # ZIP codes
        (r"\d{5}", "a 5-digit ZIP code"),
        (r"\d{5}-\d{4}", "a ZIP+4 code"),
        (r"\d{5}(?:-\d{4})?", "a ZIP code with optional +4"),
        # Hex colors
        (r"#[0-9A-Fa-f]{6}", "a 6-digit hex color like #FF5733"),
        (r"#[0-9A-Fa-f]{3,6}", "a 3 or 6 digit hex color"),
        # UUIDs
        (r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "a UUID"),
        # File paths
        (r"/(?:[^/]+/)*[^/]+", "a Unix file path"),
        (r"[A-Za-z]:\\(?:[^\\]+\\)*[^\\]+", "a Windows file path"),
        # Usernames
        (r"@[a-zA-Z0-9_]+", "a Twitter/social media handle"),
        (r"[a-zA-Z][a-zA-Z0-9_]{2,15}", "a username (3-16 chars, starts with letter)"),
        # HTML tags
        (r"<[^>]+>", "an HTML tag"),
        (r"<(\w+)[^>]*>.*?</\1>", "a matched HTML tag pair"),
        (r"<!--.*?-->", "an HTML comment"),
        # Version numbers
        (r"\d+\.\d+\.\d+", "a semantic version like 1.2.3"),
        (r"v?\d+\.\d+(?:\.\d+)?", "a version number with optional 'v' prefix"),
        # Currency
        (r"\$\d+(?:,\d{3})*(?:\.\d{2})?", "a US dollar amount like $1,234.56"),
        (r"[€£¥]\d+(?:[.,]\d+)?", "a currency amount with symbol"),
        # Percentages
        (r"\d+(?:\.\d+)?%", "a percentage like 50% or 3.14%"),
        # Log levels
        (r"\[(?:DEBUG|INFO|WARN|ERROR|FATAL)\]", "a log level tag"),
        (r"(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)", "Python logging levels"),
    ]

    for pattern, desc in patterns:
        examples.append({"prompt": f"Write a regex to match {desc}", "completion": pattern})

    return examples


def generate_explanations() -> list[dict]:
    """Explain what various regexes match."""
    examples = []

    explain_patterns = [
        (r"^\d+$", "This matches a string consisting entirely of one or more digits"),
        (r"^[a-z]+$", "This matches a string of only lowercase letters"),
        (r"\b\w{5}\b", "This matches exactly 5-character words"),
        (r".*@.*\..*", "This is a very basic email pattern - any text, @, any text, dot, any text"),
        (r"(.)\\1+", "This matches any character repeated two or more times consecutively"),
        (
            r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).{8,}$",
            "This validates a password with at least 8 chars, one uppercase, one lowercase, and one digit",
        ),
        (
            r"\b[A-Z][a-z]*\b",
            "This matches capitalized words (start with uppercase, rest lowercase)",
        ),
        (r"[^,]+", "This matches any sequence of characters that doesn't include a comma"),
        (r"(?<=@)\w+", "This matches word characters that come after an @ symbol"),
        (r"\d{3}(?=\d{4})", "This matches 3 digits only if followed by 4 more digits"),
    ]

    for pattern, explanation in explain_patterns:
        examples.append(
            {"prompt": f"Explain what this regex matches: {pattern}", "completion": explanation}
        )
        examples.append({"prompt": f"What does the regex {pattern} do?", "completion": explanation})

    return examples


def generate_fixes_and_improvements() -> list[dict]:
    """Fix broken or inefficient regexes."""
    examples = []

    fixes = [
        {
            "prompt": "Fix this regex for matching email: .*@.*",
            "completion": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,} - the original is too permissive and would match invalid emails",
        },
        {
            "prompt": "This regex <.*> matches too much in '<b>text</b>'. How do I fix it?",
            "completion": "Use a lazy quantifier <.*?> or a negated class <[^>]*> to match only until the first >",
        },
        {
            "prompt": "Why doesn't [0-9]+ match '42' in 'price: 42'?",
            "completion": r"[0-9]+ should match '42'. If it's not, check if you're using fullmatch() instead of search(). Use \d+ as shorthand.",
        },
        {
            "prompt": "My regex 'cat' is matching 'concatenate'. How do I match only the word 'cat'?",
            "completion": r"Use word boundaries: \bcat\b",
        },
        {
            "prompt": "How do I make \\d{3}-\\d{4} also match numbers with spaces like '555 1234'?",
            "completion": r"\d{3}[-\s]\d{4} - use a character class to allow either dash or space",
        },
        {
            "prompt": "My regex [a-Z] doesn't work. What's wrong?",
            "completion": "[a-Z] is invalid because 'a' to 'Z' isn't a valid range. Use [a-zA-Z] for all letters",
        },
        {
            "prompt": "How do I match a literal dot in a filename like 'file.txt'?",
            "completion": r"Escape the dot: file\.txt - an unescaped . matches any character",
        },
        {
            "prompt": "Why does my regex (a|b|c) work but is inefficient?",
            "completion": "Use a character class [abc] instead - it's more efficient for single characters",
        },
    ]

    examples.extend(fixes)
    return examples


def generate_matching_exercises() -> list[dict]:
    """Which strings match this pattern?"""
    examples = []

    exercises = [
        {
            "prompt": "Which of these match the regex \\d{3}: '123', '12', '1234', 'abc'",
            "completion": "'123' matches (exactly 3 digits). '12' has only 2 digits. '1234' has 4 digits. 'abc' has no digits.",
        },
        {
            "prompt": "Which strings match ^a.*z$: 'az', 'abz', 'a123z', 'Abz', 'azz'",
            "completion": "'az', 'abz', 'a123z', and 'azz' all match (start with 'a', end with 'z'). 'Abz' doesn't match because it starts with uppercase 'A'",
        },
        {
            "prompt": "Does 'hello world' match \\bworld\\b?",
            "completion": "Yes, 'world' appears as a complete word with word boundaries on both sides",
        },
        {
            "prompt": "Does 'color' match colou?r?",
            "completion": "Yes, 'color' matches. The u? means the 'u' is optional, so both 'color' and 'colour' match",
        },
        {
            "prompt": "Which match a+b: 'ab', 'aab', 'aaab', 'b', 'aaa'",
            "completion": "'ab', 'aab', and 'aaab' all match (one or more 'a' followed by 'b'). 'b' has no 'a'. 'aaa' has no 'b'",
        },
    ]

    examples.extend(exercises)
    return examples


def generate_extraction_tasks() -> list[dict]:
    """Extract specific parts from text."""
    examples = []

    tasks = [
        {
            "prompt": "Write a regex to extract the domain from an email address",
            "completion": r"(?<=@)[a-zA-Z0-9.-]+ or use a capturing group: @([a-zA-Z0-9.-]+)",
        },
        {"prompt": "Write a regex to extract numbers from a string", "completion": r"\d+"},
        {
            "prompt": "Write a regex to extract words from a sentence",
            "completion": r"\b\w+\b or \w+",
        },
        {
            "prompt": "Write a regex to extract the file extension from a filename",
            "completion": r"\.([a-zA-Z0-9]+)$ - the extension is in group 1",
        },
        {
            "prompt": "Write a regex to extract the path from a URL",
            "completion": r"https?://[^/]+(/[^\s?#]*)",
        },
        {"prompt": "Write a regex to extract hashtags from a tweet", "completion": r"#\w+"},
        {"prompt": "Write a regex to extract all @mentions from text", "completion": r"@\w+"},
        {
            "prompt": "Write a regex to extract the year from a date like '2024-01-15'",
            "completion": r"^(\d{4})-\d{2}-\d{2}$ - year is in group 1",
        },
        {
            "prompt": "Write a regex to extract content between square brackets",
            "completion": r"\[([^\]]+)\]",
        },
        {
            "prompt": "Write a regex to extract quoted strings",
            "completion": r'"([^"]*)"' + " or " + r"'([^']*)'",
        },
    ]

    examples.extend(tasks)
    return examples


def generate_validation_patterns() -> list[dict]:
    """Patterns for validating input."""
    examples = []

    validations = [
        {
            "prompt": "Write a regex to validate a username (3-16 alphanumeric characters, starting with a letter)",
            "completion": r"^[a-zA-Z][a-zA-Z0-9_]{2,15}$",
        },
        {
            "prompt": "Write a regex to validate a strong password (8+ chars, uppercase, lowercase, digit, special char)",
            "completion": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$",
        },
        {
            "prompt": "Write a regex to validate a hex color code",
            "completion": r"^#[0-9A-Fa-f]{6}$",
        },
        {"prompt": "Write a regex to validate a positive integer", "completion": r"^[1-9]\d*$"},
        {"prompt": "Write a regex to validate a decimal number", "completion": r"^-?\d+\.?\d*$"},
        {
            "prompt": "Write a regex to validate a slug (lowercase letters, numbers, hyphens)",
            "completion": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        },
        {
            "prompt": "Write a regex to validate a MAC address",
            "completion": r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$",
        },
        {
            "prompt": "Write a regex to validate a time in 24-hour format (HH:MM)",
            "completion": r"^([01]\d|2[0-3]):[0-5]\d$",
        },
        {
            "prompt": "Write a regex to validate a date in YYYY-MM-DD format",
            "completion": r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$",
        },
        {
            "prompt": "Write a regex to validate an IPv4 address",
            "completion": r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$",
        },
    ]

    examples.extend(validations)
    return examples


def generate_replacement_examples() -> list[dict]:
    """Search and replace patterns."""
    examples = []

    replacements = [
        {
            "prompt": "How do I replace all digits with X using regex?",
            "completion": r"Use pattern \d and replacement 'X', or \d+ to replace sequences of digits",
        },
        {
            "prompt": "How do I remove all HTML tags from text?",
            "completion": r"Match <[^>]+> and replace with empty string",
        },
        {
            "prompt": "How do I swap first and last name in 'Doe, John' to 'John Doe'?",
            "completion": r"Match (\w+), (\w+) and replace with \2 \1",
        },
        {
            "prompt": "How do I add quotes around words?",
            "completion": r"Match \b(\w+)\b and replace with '\1' or \"\1\"",
        },
        {
            "prompt": "How do I normalize whitespace (multiple spaces to single)?",
            "completion": r"Match \s+ and replace with a single space",
        },
        {
            "prompt": "How do I convert dates from MM/DD/YYYY to YYYY-MM-DD?",
            "completion": r"Match (\d{2})/(\d{2})/(\d{4}) and replace with \3-\1-\2",
        },
        {
            "prompt": "How do I remove duplicate consecutive words?",
            "completion": r"Match \b(\w+)\s+\1\b and replace with \1",
        },
        {
            "prompt": "How do I convert camelCase to snake_case?",
            "completion": r"Match ([a-z])([A-Z]) and replace with \1_\2, then lowercase",
        },
    ]

    examples.extend(replacements)
    return examples


def generate_all() -> list[dict]:
    """Generate all training examples."""
    all_examples = []

    generators = [
        generate_literal_matching,
        generate_escaping,
        generate_character_classes,
        generate_predefined_classes,
        generate_anchors,
        generate_quantifiers,
        generate_greedy_vs_lazy,
        generate_alternation,
        generate_grouping,
        generate_backreferences,
        generate_lookahead,
        generate_lookbehind,
        generate_common_patterns,
        generate_explanations,
        generate_fixes_and_improvements,
        generate_matching_exercises,
        generate_extraction_tasks,
        generate_validation_patterns,
        generate_replacement_examples,
    ]

    for gen_func in generators:
        examples = gen_func()
        all_examples.extend(examples)
        print(f"  {gen_func.__name__}: {len(examples)} examples")

    return all_examples


def augment_with_variations(examples: list[dict], target_count: int) -> list[dict]:
    """Add variations to reach target count."""
    augmented = list(examples)
    rng = random.Random(42)

    # Prompt variations
    prompt_prefixes = [
        "Write a regex to match ",
        "Create a regex pattern for ",
        "What regex matches ",
        "Give me a regex for ",
        "How do I write a regex to match ",
        "I need a regex pattern that matches ",
        "Construct a regex for ",
        "Provide a regex to find ",
    ]

    prompt_suffixes = [
        "",
        " Please be precise.",
        " Keep it simple.",
        " Use standard regex syntax.",
    ]

    # Generate variations until we hit target
    while len(augmented) < target_count:
        # Pick a random existing example to vary
        ex = rng.choice(examples)
        prompt = ex["prompt"]
        completion = ex["completion"]

        # Create variation
        if prompt.startswith("Write a regex"):
            # Replace with alternative phrasing
            rest = prompt[len("Write a regex") :]
            new_prefix = rng.choice(prompt_prefixes)
            new_suffix = rng.choice(prompt_suffixes)

            # Extract the description part
            if " to match " in rest:
                desc = rest.split(" to match ", 1)[1]
            elif " that matches " in rest:
                desc = rest.split(" that matches ", 1)[1]
            elif " for " in rest:
                desc = rest.split(" for ", 1)[1]
            else:
                desc = rest.strip()

            new_prompt = f"{new_prefix}{desc}{new_suffix}"
            augmented.append({"prompt": new_prompt, "completion": completion})
        elif prompt.startswith("What does"):
            # Add "Explain" variation
            augmented.append(
                {"prompt": prompt.replace("What does", "Explain what"), "completion": completion}
            )
        else:
            # Just add with minor rephrasing
            augmented.append({"prompt": f"Regex question: {prompt}", "completion": completion})

    return augmented[:target_count]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate regex SFT training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("config/training/regex_sft_data.jsonl"),
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--count", type=int, default=3000, help="Target number of examples (will augment if needed)"
    )
    args = parser.parse_args()

    print("Generating regex training examples...")
    examples = generate_all()
    print(f"Generated {len(examples)} base examples")

    if len(examples) < args.count:
        print(f"Augmenting to {args.count} examples...")
        examples = augment_with_variations(examples, args.count)

    # Shuffle
    random.seed(42)
    random.shuffle(examples)

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
