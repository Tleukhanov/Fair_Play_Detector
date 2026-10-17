"""Add versioned analysis cache identity."""

import sqlalchemy as sa
from alembic import op

revision = "0004_analysis_cache"
down_revision = "0003_verdict_confidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("game_features", sa.Column("cache_key", sa.String(128), nullable=True))
    op.add_column("game_features", sa.Column("pgn_hash", sa.String(64), nullable=True))
    op.create_index("ix_game_features_cache_key", "game_features", ["cache_key"], unique=True)
    op.create_index("ix_game_features_pgn_hash", "game_features", ["pgn_hash"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_game_features_pgn_hash", table_name="game_features")
    op.drop_index("ix_game_features_cache_key", table_name="game_features")
    op.drop_column("game_features", "pgn_hash")
    op.drop_column("game_features", "cache_key")
