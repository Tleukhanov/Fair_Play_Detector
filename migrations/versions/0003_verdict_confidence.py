"""Store the confidence interval used by the verdict contract."""

import sqlalchemy as sa
from alembic import op

revision = "0003_verdict_confidence"
down_revision = "0002_scan_lease"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("verdicts", sa.Column("confidence_lower", sa.Float(), nullable=True))
    op.add_column("verdicts", sa.Column("confidence_upper", sa.Float(), nullable=True))
    op.execute(
        "UPDATE verdicts SET confidence_lower = risk_score, confidence_upper = risk_score"
    )
    with op.batch_alter_table("verdicts") as batch:
        batch.alter_column("confidence_lower", nullable=False)
        batch.alter_column("confidence_upper", nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("verdicts") as batch:
        batch.drop_column("confidence_upper")
        batch.drop_column("confidence_lower")
