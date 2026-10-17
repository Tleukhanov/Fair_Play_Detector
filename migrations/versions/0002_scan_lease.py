"""Add a unique worker lease to scans."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_scan_lease"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("scans", sa.Column("lease_id", sa.String(36), nullable=True))
    op.create_index("ix_scans_lease_id", "scans", ["lease_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_scans_lease_id", table_name="scans")
    op.drop_column("scans", "lease_id")
