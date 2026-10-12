"""Unit tests for the scorer in src/evaluate.py. Run: python tests/test_evaluate.py (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluate import extract_sql, is_format_ok, is_valid, normalize  # noqa: E402

SCHEMA = "CREATE TABLE singer (name VARCHAR, age INTEGER)"


def test_extract_sql_strips_fence_and_semicolon():
    assert extract_sql("```sql\nSELECT name FROM singer;\n```") == "SELECT name FROM singer"
    assert extract_sql("SELECT 1;") == "SELECT 1"


def test_format_ok_accepts_bare_query_only():
    assert is_format_ok("SELECT name FROM singer")
    assert is_format_ok("WITH t AS (SELECT 1) SELECT * FROM t")
    assert not is_format_ok("```sql\nSELECT 1\n```")
    assert not is_format_ok("SELECT 1\n\nThis query returns one.")
    assert not is_format_ok("Here is the query: SELECT 1")


def test_normalize_ignores_spacing_case_and_quoted_numbers():
    assert normalize("select  name from singer where age = 30") == normalize("SELECT name FROM singer WHERE age = '30'")
    assert normalize('SELECT "name" FROM singer') == normalize("SELECT 'name' FROM singer")


def test_normalize_distinguishes_different_queries():
    assert normalize("SELECT name FROM singer") != normalize("SELECT age FROM singer")
    assert normalize("SELECT name FROM singer WHERE age > 30") != normalize("SELECT name FROM singer WHERE age < 30")


def test_normalize_returns_none_for_unparseable_sql():
    assert normalize("SELEC FROM WHERE ((") is None


def test_is_valid_checks_sql_against_schema():
    assert is_valid(SCHEMA, "SELECT name FROM singer")
    assert not is_valid(SCHEMA, "SELECT nope FROM singer")
    assert not is_valid(SCHEMA, "SELECT name FROM missing_table")
    assert not is_valid(SCHEMA, "this is not sql")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
