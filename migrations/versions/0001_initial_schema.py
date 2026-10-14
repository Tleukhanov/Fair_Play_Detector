"""Create initial Fair Play Detector schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("lichess_username", sa.String(30), nullable=False),
        sa.Column("username_normalized", sa.String(30), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("username_normalized"),
    )
    op.create_index("ix_users_username_normalized", "users", ["username_normalized"], unique=False)
    op.create_table(
        "scans",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("requested_time_control", sa.String(20), nullable=False),
        sa.Column("requested_game_limit", sa.Integer(), nullable=False),
        sa.Column("games_found", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("games_analyzed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model_version", sa.String(100)),
        sa.Column("error_code", sa.String(100)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("completed_at", sa.DateTime()),
    )
    op.create_index("ix_scans_user_id", "scans", ["user_id"])
    op.create_table(
        "games",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("lichess_game_id", sa.String(20), nullable=False),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("scan_id", sa.String(36), sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("played_at", sa.DateTime(), nullable=False),
        sa.Column("rated", sa.Boolean(), nullable=False),
        sa.Column("speed", sa.String(20), nullable=False),
        sa.Column("variant", sa.String(30), nullable=False),
        sa.Column("white_username", sa.String(30), nullable=False),
        sa.Column("black_username", sa.String(30), nullable=False),
        sa.Column("user_color", sa.String(5), nullable=False),
        sa.Column("result", sa.String(7), nullable=False),
        sa.Column("opening", sa.String(100)),
        sa.Column("pgn", sa.Text()),
        sa.Column("pgn_hash", sa.String(64), nullable=False),
        sa.Column("source_url", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("lichess_game_id", "user_id"),
    )
    op.create_index("ix_games_user_id", "games", ["user_id"])
    op.create_index("ix_games_scan_id", "games", ["scan_id"])
    op.create_table(
        "game_features",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("game_id", sa.String(36), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("feature_version", sa.String(30), nullable=False),
        sa.Column("aggregates", sa.JSON(), nullable=False),
        sa.Column("moves", sa.JSON(), nullable=False),
        sa.Column("engine_name", sa.String(50)),
        sa.Column("engine_version", sa.String(50)),
        sa.Column("depth_or_time_limit", sa.String(50)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_game_features_game_id", "game_features", ["game_id"])
    op.create_table(
        "game_scores",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("game_id", sa.String(36), sa.ForeignKey("games.id"), nullable=False),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("top_move_match_rate", sa.Float(), nullable=False),
        sa.Column("mean_centipawn_loss", sa.Float(), nullable=False),
        sa.Column("maia_divergence", sa.Float()),
        sa.Column("explainability", sa.JSON(), nullable=False),
    )
    op.create_index("ix_game_scores_game_id", "game_scores", ["game_id"])
    op.create_table(
        "verdicts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("scan_id", sa.String(36), sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("risk_level", sa.String(20), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("limitations", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(100)),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("scan_id"),
    )


def downgrade() -> None:
    op.drop_table("verdicts")
    op.drop_table("game_scores")
    op.drop_table("game_features")
    op.drop_table("games")
    op.drop_table("scans")
    op.drop_index("ix_users_username_normalized", table_name="users")
    op.drop_table("users")
