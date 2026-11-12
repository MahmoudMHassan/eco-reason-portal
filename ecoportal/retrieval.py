"""Local dense and BM25 passage search over a versioned Zenodo snapshot."""

import argparse
import hashlib
import os
from contextlib import closing
from functools import cached_property
from itertools import batched
from pathlib import Path
from time import sleep
from urllib.error import URLError
from urllib.request import urlopen
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from .corpus import Corpus, ensure_bundled_corpus

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
INDEX_VERSION = "v2"
MISSING = {"", "n/a", "na", "null", "nan", "none"}


def passages(corpus):
    with corpus.connect() as db:
        for row in db.execute("SELECT * FROM papers ORDER BY paper_id"):
            for kind, raw in (
                ("Abstract", row["abstract"]),
                ("Full text", row["full_text"]),
            ):
                if raw.strip().casefold() in MISSING:
                    continue
                step = len(raw) if kind == "Abstract" else 2400
                for start in range(0, len(raw), step):
                    excerpt = (
                        raw[start:] if kind == "Abstract" else raw[start : start + 2600]
                    )
                    if not excerpt.strip():
                        continue
                    yield {
                        "id": f"{row['paper_id']}:{kind}:{start}",
                        "paper_id": row["paper_id"],
                        "doi": row["doi"],
                        "title": row["title"],
                        "tags": row["tags"],
                        "locator": f"{kind}, characters {start}–{start + len(excerpt)}",
                        "text": excerpt,
                    }


def sparse_vector(embedding):
    return models.SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.tolist()
    )


class HybridSearch:
    def __init__(self, corpus, client):
        self.corpus = corpus
        self.client = client

    @cached_property
    def collection(self):
        with self.corpus.path.open("rb") as snapshot:
            fingerprint = hashlib.file_digest(snapshot, "sha256").hexdigest()[:16]
        return f"ecoreason_{INDEX_VERSION}_{fingerprint}"

    @cached_property
    def dense(self):
        from fastembed import TextEmbedding

        return TextEmbedding(DENSE_MODEL, threads=8)

    @cached_property
    def sparse(self):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(SPARSE_MODEL, threads=2)

    def build(self):
        if self.client.collection_exists(self.collection):
            return self.client.count(self.collection, exact=True).count

        staging = f"{self.collection}_build"
        if not self.client.collection_exists(staging):
            self.client.create_collection(
                staging,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        count = 0
        for batch in batched(passages(self.corpus), 192):
            # One dense representation per paper. BM25 covers the full text of every passage.
            abstracts = [row for row in batch if row["locator"].startswith("Abstract")]
            dense_by_id = (
                {
                    row["id"]: dense.tolist()
                    for row, dense in zip(
                        abstracts,
                        self.dense.passage_embed(
                            [
                                f"{row['title']}\n{row['text'][:600]}"
                                for row in abstracts
                            ],
                            batch_size=32,
                        ),
                        strict=True,
                    )
                }
                if abstracts
                else {}
            )
            sparse_text = [
                f"{row['title']}\n{row['text']}\n{row['tags'] if row['id'] in dense_by_id else ''}"
                for row in batch
            ]
            points = []
            for row, sparse in zip(batch, self.sparse.embed(sparse_text), strict=True):
                vectors = {"bm25": sparse_vector(sparse)}
                if row["id"] in dense_by_id:
                    vectors["dense"] = dense_by_id[row["id"]]
                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, row["id"])),
                        vector=vectors,
                        payload=row,
                    )
                )
            self.client.upsert(staging, points, wait=True)
            count += len(points)
            if count % 1920 == 0:
                print(f"Indexed {count:,} passages", flush=True)
        self.client.update_collection_aliases(
            [
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=staging, alias_name=self.collection
                    )
                )
            ]
        )
        return count

    def search(self, query, limit):
        if query.strip().lower().startswith(("10.", "https://doi.org/", "doi:")):
            return None
        try:
            if not self.client.collection_exists(self.collection):
                return None
            dense = next(iter(self.dense.query_embed(query))).tolist()
            sparse = sparse_vector(next(iter(self.sparse.query_embed(query))))
            # Overfetch passages so multiple matches from one paper don't crowd out others.
            pool = limit * 6
            results = self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using="dense", limit=pool),
                    models.Prefetch(query=sparse, using="bm25", limit=pool),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=pool,
            )
        except (ResponseHandlingException, UnexpectedResponse, ValueError):
            return None

        hits = {}
        for point in results.points:
            row = point.payload
            if row["paper_id"] not in hits:
                hits[row["paper_id"]] = {**row, "abstract": row["text"]}
                if len(hits) == limit:
                    break
        return list(hits.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.getenv("ECO_DATA_DIR", "runtime"))
    )
    parser.add_argument(
        "--initialize", action="store_true", help="Import the bundled corpus if needed."
    )
    args = parser.parse_args()
    corpus = Corpus(args.data_dir / "corpus.sqlite")
    qdrant_url = os.getenv("ECO_QDRANT_URL", "http://localhost:6333")
    if args.initialize:
        ensure_bundled_corpus(corpus.path)
        for attempt in range(30):
            try:
                with urlopen(f"{qdrant_url.rstrip('/')}/readyz", timeout=2):
                    break
            except (URLError, TimeoutError):
                if attempt == 29:
                    raise
                sleep(2)
    if not corpus.path.is_file():
        parser.error("Import the corpus before building its Qdrant index.")
    with closing(QdrantClient(url=qdrant_url, timeout=30)) as client:
        search = HybridSearch(corpus, client)
        print(f"{search.collection}: {search.build():,} passages")


if __name__ == "__main__":
    main()
