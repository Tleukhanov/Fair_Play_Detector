"""Per-scan game snapshots, reusable feature cache, and scan options.

The upgrade rewrites three uniqueness rules that blocked re-scans and cache
reuse:

* ``games`` becomes a per-scan snapshot: ``(lichess_game_id, scan_id)`` is
  unique instead of ``(lichess_game_id, user_id)``;
* ``game_features.cache_key`` stops being globally unique and
  ``(game_id, cache_key)`` becomes unique so cached analysis can be copied onto
  a new game row;
* ``game_scores`` gets a ``(game_id, model_version)`` uniqueness rule so
  re-scoring replaces instead of duplicates rows.

SQLite needs table rebuilds for constraint changes, so all constraint edits go
through ``batch_alter_table``.  Existing rows are de-duplicated first, keeping
the earliest row of each group, which makes both directions reversible.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_scan_pipeline_fixes"
down_revision: str | None = "0004_analysis_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Names assigned to the SQLite table-level UNIQUE constraints.  The games
# constraint created in 0001 is unnamed, so batch reflection needs a naming
# convention to address it during the downgrade/upgrade.
_NAMING_CONVENTION = {"uq": "uq_%(table_name)s_%(column_0_name)s_%(column_1_name)s"}
_UQ_GAMES_BY_USER = "uq_games_lichess_game_id_user_id"
_UQ_GAMES_BY_SCAN = "uq_games_lichess_game_id_scan_id"
_UQ_FEATURES_GAME_CACHE = "uq_game_features_game_id_cache_key"
_UQ_SCORES_GAME_MODEL = "uq_game_scores_game_id_model_version"


def upgrade() -> None:
    op.add_column(
        "scans",
        sa.Column("rated_only", sa.Boolean(), nullable=False, server_default=sa.true()),
    )

    # A re-run of the scoring stage may already have produced duplicate rows.
    op.execute(
        sa.text(
            "DELETE FROM game_scores WHERE id NOT IN ("
            "SELECT MIN(id) FROM game_scores GROUP BY game_id, model_version)"
        )
    )
    with op.batch_alter_table("game_scores") as batch:
        batch.create_unique_constraint(_UQ_SCORES_GAME_MODEL, ["game_id", "model_version"])

    # cache_key was a unique index from 0004; it must become a plain index.
    op.drop_index("ix_game_features_cache_key", table_name="game_features")
    op.create_index("ix_game_features_cache_key", "game_features", ["cache_key"], unique=False)
    with op.batch_alter_table("game_features") as batch:
        batch.create_unique_constraint(_UQ_FEATURES_GAME_CACHE, ["game_id", "cache_key"])

    with op.batch_alter_table("games", naming_convention=_NAMING_CONVENTION) as batch:
        batch.drop_constraint(_UQ_GAMES_BY_USER, type_="unique")
        batch.create_unique_constraint(_UQ_GAMES_BY_SCAN, ["lichess_game_id", "scan_id"])


def downgrade() -> None:
    # Re-scans create one game row per scan for the same Lichess game, which the
    # pre-0005 unique rule cannot represent.  Keep the earliest row per user.
    op.execute(
        sa.text(
            "DELETE FROM game_features WHERE game_id NOT IN ("
            "SELECT MIN(id) FROM games GROUP BY lichess_game_id, user_id)"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM game_scores WHERE game_id NOT IN ("
            "SELECT MIN(id) FROM games GROUP BY lichess_game_id, user_id)"
        )
    )
    op.execute(
        sa.text(
            "DELETE FROM games WHERE id NOT IN ("
            "SELECT MIN(id) FROM games GROUP BY lichess_game_id, user_id)"
        )
    )
    with op.batch_alter_table("games", naming_convention=_NAMING_CONVENTION) as batch:
        batch.drop_constraint(_UQ_GAMES_BY_SCAN, type_="unique")
        batch.create_unique_constraint(_UQ_GAMES_BY_USER, ["lichess_game_id", "user_id"])

    with op.batch_alter_table("game_features") as batch:
        batch.drop_constraint(_UQ_FEATURES_GAME_CACHE, type_="unique")
    # A copy-on-cache-hit row may share its cache_key with the original row.
    op.execute(
        sa.text(
            "DELETE FROM game_features WHERE id NOT IN ("
            "SELECT MIN(id) FROM game_features GROUP BY cache_key)"
        )
    )
    op.drop_index("ix_game_features_cache_key", table_name="game_features")
    op.create_index("ix_game_features_cache_key", "game_features", ["cache_key"], unique=True)

    with op.batch_alter_table("game_scores") as batch:
        batch.drop_constraint(_UQ_SCORES_GAME_MODEL, type_="unique")

    with op.batch_alter_table("scans") as batch:
        batch.drop_column("rated_only")
