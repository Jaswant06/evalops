-- Runs once, when the database volume is first created.
-- MLflow keeps its tracking and registry tables in their own database.
CREATE DATABASE mlflow;

-- Grafana reads results with a read-only role; it can never change them.
CREATE ROLE grafana_reader LOGIN PASSWORD 'grafana_reader';
GRANT CONNECT ON DATABASE evalops TO grafana_reader;
GRANT USAGE ON SCHEMA public TO grafana_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE evalops IN SCHEMA public GRANT SELECT ON TABLES TO grafana_reader;
