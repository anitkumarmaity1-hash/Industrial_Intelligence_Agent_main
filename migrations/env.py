"""Alembic environment — raw-SQL migrations (this project has no ORM models)."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import create_engine, pool

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False: fileConfig's default (True) disables
    # every logger not named in alembic.ini's [loggers] section (e.g.
    # app.agents.planner), which broke caplog-based tests elsewhere in the
    # suite whenever they ran after this module was imported.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

load_dotenv()


def _database_url() -> str:
    # Tests / callers may pass an explicit URL: alembic -x url=... or
    # config.set_main_option("sqlalchemy.url", ...).
    url = (context.get_x_argument(as_dictionary=True).get("url")
           or config.get_main_option("sqlalchemy.url")
           or os.environ.get("DATABASE_URL"))
    if not url:
        raise RuntimeError("DATABASE_URL is not set (see .env.example).")
    return url


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=None, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
