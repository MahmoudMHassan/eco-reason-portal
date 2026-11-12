"""Persistent claim versions and decisions, scoped to a local workspace."""

import hashlib
import json
from pathlib import Path

import aiosqlite


class ReviewStore:
    def __init__(self, path):
        self.path = Path(path)

    async def setup(self):
        async with aiosqlite.connect(self.path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS claim_versions (
                    workspace TEXT NOT NULL,
                    claim_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(workspace, claim_id, version)
                )
            """)
            await db.commit()

    async def get(self, workspace):
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT * FROM claim_versions AS v
                WHERE workspace = ? AND version = (
                    SELECT max(version) FROM claim_versions
                    WHERE workspace = v.workspace AND claim_id = v.claim_id
                )
                ORDER BY created_at, claim_id
            """,
                (workspace,),
            ) as cursor:
                return [
                    {
                        **json.loads(row["payload"]),
                        **{
                            key: row[key]
                            for key in (
                                "claim_id",
                                "version",
                                "status",
                                "scope",
                                "reason",
                            )
                        },
                    }
                    for row in await cursor.fetchall()
                ]

    async def propose(self, workspace, claims):
        async with aiosqlite.connect(self.path) as db:
            for claim in claims:
                raw = json.dumps(claim, sort_keys=True, ensure_ascii=False)
                claim_id = hashlib.sha256(raw.encode()).hexdigest()[:20]
                await db.execute(
                    """
                    INSERT OR IGNORE INTO claim_versions
                    (workspace, claim_id, version, status, scope, reason, payload)
                    VALUES (?, ?, 1, 'proposed', ?, '', ?)
                """,
                    (workspace, claim_id, claim["scope"], raw),
                )
            await db.commit()
        return await self.get(workspace)

    async def review(self, workspace, claim_id, version, action, scope="", reason=""):
        if action not in {"accepted", "revised", "rejected"}:
            raise ValueError("Unknown review action.")
        if action in {"revised", "rejected"} and not reason.strip():
            raise ValueError("Give a reason for the revision or rejection.")
        if action == "revised" and not scope.strip():
            raise ValueError("Enter the revised scope.")
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                """
                SELECT * FROM claim_versions WHERE workspace = ? AND claim_id = ?
                ORDER BY version DESC LIMIT 1
            """,
                (workspace, claim_id),
            ) as cursor:
                row = await cursor.fetchone()
            if not row or row["version"] != version:
                raise ValueError(
                    "This claim changed. Refresh the review and try again."
                )
            await db.execute(
                """
                INSERT INTO claim_versions
                (workspace, claim_id, version, status, scope, reason, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    workspace,
                    claim_id,
                    version + 1,
                    action,
                    scope.strip() if action == "revised" else row["scope"],
                    reason.strip(),
                    row["payload"],
                ),
            )
            await db.commit()
        return await self.get(workspace)
