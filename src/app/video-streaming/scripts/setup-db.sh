#!/bin/bash

set -e

echo "========================================"
echo "Setting up PostgreSQL Database"
echo "========================================"

DB_USER=${POSTGRES_USER:-postgres}
DB_NAME=${POSTGRES_DB:-streaming_db}
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5432}

echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo "Host: $DB_HOST:$DB_PORT"
echo ""

# Check if database exists
if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -lqt | cut -d \| -f 1 | grep -qw $DB_NAME; then
    echo "Database '$DB_NAME' already exists"
else
    echo "Creating database '$DB_NAME'..."
    createdb -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME
    echo "Database created successfully"
fi

# Run migrations
echo ""
echo "Running migrations..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f scripts/migrate.sql

echo ""
echo "Database setup complete!"