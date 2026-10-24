"""Persist the structured explanation produced by the explaining stage.

The numeric verdict is created by the scoring stage with a deterministic initial
explanation.  The fourth pipeline stage then replaces it with either a grounded
LLM explanation or the deterministic fallback.  Both new columns are nullable so
verdicts written before this migration remain readable.
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_llm_explanation"
down_revision = "0005_scan_pipeline_fixes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("verdicts", sa.Column("thresholds", sa.JSON(), nullable=True))
    op.add_column("verdicts", sa.Column("explanation", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("verdicts", "explanation")
    op.drop_column("verdicts", "thresholds")
