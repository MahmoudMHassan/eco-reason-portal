"""Build and query a DOI-joined SQLite index of Zenodo release 14525619."""

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing
from lzma import open as open_xz
from pathlib import Path
from urllib.parse import quote, unquote
from urllib.request import urlopen
from zipfile import ZipFile

RECORD = "https://zenodo.org/records/14525619"
FILES = {
    "publications_tdm.csv": "b200dd1e1518a4778c0ad4c16159e1d7",
    "structured-publications.zip": "f1ab504e62b88db821dd56b40de157b9",
}
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "studies.json"
EXAMPLE_STUDIES = {paper["key"]: paper for paper in json.loads(EXAMPLES.read_text())}
BUNDLED_DATA = EXAMPLES.parent.parent / "data"
ZENODO_SNAPSHOT = "Zenodo v3"


def normalize_doi(value):
    value = unquote(value or "").strip().lower()
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)


def doi_url(doi):
    return "https://doi.org/" + quote(doi, safe="/") if doi else ""


def extract_tags(archive):
    tags = {}
    with ZipFile(archive) as z:
        for name in z.namelist():
            if not name.lower().endswith(".json") or not name.startswith("10."):
                continue
            try:
                data = json.loads(z.read(name))
            except (ValueError, UnicodeError):
                continue
            if not isinstance(data, dict):
                continue
            terms = []
            for field in (
                "species",
                "location",
                "habitat",
                "ecosystem",
                "relationships",
            ):
                values = data.get(field)
                if not isinstance(values, list):
                    continue
                for value in values:
                    if isinstance(value, dict):
                        label = value.get("name")
                        props = value.get("relationship_properties")
                        if isinstance(props, dict):
                            label = props.get("name", label)
                        if isinstance(label, str):
                            terms.append(label)
            tags[normalize_doi(name[:-5].replace("_", "/", 1))] = " ".join(
                dict.fromkeys(terms)
            )
    return tags


def publications(path, tags):
    csv.field_size_limit(64 * 1024 * 1024)
    opener = open_xz if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"ASK ID", "DOI", "Title", "Abstract", "Full-text"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unexpected Zenodo CSV columns")
        for row in reader:
            doi = normalize_doi(row["DOI"])
            ask_id = (row["ASK ID"] or "").strip()
            if doi or ask_id:
                yield (
                    doi or f"ask:{ask_id}",
                    doi,
                    row["Title"] or "",
                    row["Abstract"] or "",
                    row["Full-text"] or "",
                    tags.get(doi, ""),
                )


def build_index(path, csv_path=None, archive=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tags = extract_tags(archive) if archive else {}
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        working = Path(directory) / "index.sqlite"
        with closing(sqlite3.connect(working)) as db, db:
            db.executescript("""
                CREATE TABLE papers (
                    paper_id TEXT PRIMARY KEY, doi TEXT, title TEXT,
                    abstract TEXT, full_text TEXT, tags TEXT
                );
                CREATE VIRTUAL TABLE search_index USING fts5(
                    title, abstract, tags, content='papers', content_rowid='rowid'
                );
                CREATE TABLE snapshot (name TEXT NOT NULL);
            """)
            if csv_path:
                db.executemany(
                    "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                    publications(Path(csv_path), tags),
                )
            if not csv_path:
                db.executemany(
                    "INSERT OR IGNORE INTO papers VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            paper["doi"],
                            paper["doi"],
                            paper["title"],
                            paper["passage"],
                            "",
                            paper["tags"],
                        )
                        for paper in EXAMPLE_STUDIES.values()
                    ],
                )
            db.execute("INSERT INTO search_index(search_index) VALUES ('rebuild')")
            db.execute(
                "INSERT INTO snapshot VALUES (?)",
                (ZENODO_SNAPSHOT if csv_path else "demo excerpts only",),
            )
            stats = db.execute(
                "SELECT COUNT(*), SUM(lower(trim(full_text)) NOT IN ('', 'n/a', 'na', 'null', 'nan', 'none')), SUM(tags != '') FROM papers"
            ).fetchone()
        working.replace(path)
    return stats


STOP = set(
    "a an and are as at be by can could did do does for from had has have how i "
    "in into is it me of on or our please should study than that the their this to "
    "under was were what when where which who will with would you about look find "
    "research evidence hypothesis papers test assess studies".split()
)


class Corpus:
    def __init__(self, path):
        self.path = Path(path)

    def connect(self):
        db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        return closing(db)

    def info(self):
        with self.connect() as db:
            return {
                "snapshot": db.execute("SELECT name FROM snapshot").fetchone()[0],
                "papers": db.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
            }

    def search(self, query, limit=8):
        words = list(
            dict.fromkeys(
                token
                for token in re.findall(r"[^\W_]+", query.lower())
                if token not in STOP and len(token) > 2
            )
        )[:30]
        with self.connect() as db:
            if query.lower().strip().startswith(("10.", "https://doi.org/", "doi:")):
                direct = db.execute(
                    "SELECT paper_id, doi, title, abstract FROM papers WHERE paper_id = ?",
                    (normalize_doi(query),),
                ).fetchone()
                if direct:
                    return [dict(direct)]
            if not words:
                return []
            rows = db.execute(
                """
                SELECT p.paper_id, p.doi, p.title, p.abstract
                FROM search_index JOIN papers p ON p.rowid = search_index.rowid
                WHERE search_index MATCH ?
                ORDER BY bm25(search_index, 4.0, 1.0, 0.6), p.paper_id
                LIMIT ?
            """,
                (" OR ".join(f'"{word}"' for word in words), min(max(limit, 1), 20)),
            )
            return [dict(row) for row in rows]

    def passages(self, paper_ids, query):
        terms = set(re.findall(r"[^\W_]+", query.lower())) - STOP
        passages = []
        with self.connect() as db:
            for paper_id in dict.fromkeys(paper_ids):
                row = db.execute(
                    "SELECT doi, title, abstract, full_text FROM papers WHERE paper_id = ?",
                    (paper_id,),
                ).fetchone()
                if not row:
                    continue
                for kind, raw in (
                    ("Abstract", row["abstract"]),
                    ("Full text", row["full_text"]),
                ):
                    if raw.strip().casefold() in {
                        "",
                        "n/a",
                        "na",
                        "null",
                        "nan",
                        "none",
                    }:
                        continue
                    # Bound context while retaining a stable locator into the stored text.
                    windows = [
                        (start, raw[start : start + 2000])
                        for start in range(0, len(raw), 1750)
                    ]
                    start, excerpt = max(
                        windows,
                        key=lambda pair: len(
                            terms & set(re.findall(r"[^\W_]+", pair[1].lower()))
                        ),
                    )
                    passages.append(
                        {
                            "id": f"{paper_id}:{kind}:{start}",
                            "doi": row["doi"],
                            "title": row["title"],
                            "locator": f"{kind}, characters {start}–{start + len(excerpt)}",
                            "text": excerpt,
                        }
                    )
        return passages


def ensure_bundled_corpus(path):
    if path.is_file() and Corpus(path).info()["snapshot"] == ZENODO_SNAPSHOT:
        return

    print("Importing the bundled Zenodo corpus", flush=True)
    stats = build_index(
        path,
        BUNDLED_DATA / "publications_tdm.csv.xz",
        BUNDLED_DATA / "structured-publications.zip",
    )
    print(
        f"Imported {stats[0]:,} papers, {stats[1]:,} with full text, "
        f"{stats[2]:,} with extraction tags.",
        flush=True,
    )


def download_release(directory):
    for name, checksum in FILES.items():
        target = directory / name
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=directory) as temp:
                with urlopen(f"{RECORD}/files/{name}?download=1", timeout=60) as source:
                    shutil.copyfileobj(source, temp)
                temp.flush()
                temp.seek(0)
                if hashlib.file_digest(temp, "md5").hexdigest() != checksum:
                    raise ValueError(f"Zenodo checksum mismatch for {name}")
                shutil.copyfile(temp.name, target)
        with target.open("rb") as file:
            if hashlib.file_digest(file, "md5").hexdigest() != checksum:
                raise ValueError(f"Zenodo checksum mismatch for {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("runtime"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    if args.download:
        download_release(args.data_dir)
    csv_path = args.csv or args.data_dir / "publications_tdm.csv"
    archive = args.archive or args.data_dir / "structured-publications.zip"
    if args.csv and not csv_path.is_file():
        parser.error(f"CSV missing: {csv_path}")
    if args.archive and not archive.is_file():
        parser.error(f"Archive missing: {archive}")
    stats = build_index(
        args.data_dir / "corpus.sqlite",
        csv_path if csv_path.is_file() else None,
        archive if archive.is_file() else None,
    )
    print(
        f"Indexed {stats[0]} papers, {stats[1]} with full text, {stats[2]} with extraction tags."
    )


if __name__ == "__main__":
    main()
