-- ============================================================
-- GARIP DuckDB Schema
-- GCP equivalent: BigQuery dataset garip_regulations
-- ============================================================

-- ------------------------------------------------------------
-- Core regulations table
-- PK: regulation_id + version_id (supports amendment versioning)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS regulations (
    regulation_id           VARCHAR         NOT NULL,
    version_id              INTEGER         NOT NULL DEFAULT 1,
    source_hash             VARCHAR(64)     NOT NULL,
    title                   VARCHAR(1000),
    jurisdiction            VARCHAR(10)     NOT NULL,
    regulation_type         VARCHAR(100),
    effective_date          DATE,
    source_url              VARCHAR(2000)   NOT NULL,
    raw_text                TEXT,
    ingested_at             TIMESTAMPTZ     DEFAULT NOW(),
    source_name             VARCHAR(50),

    -- NER-enriched fields (populated by ner_tagger.py)
    article_citations       VARCHAR,        -- pipe-separated: "Article 5|Article 25"
    fine_amounts            VARCHAR,        -- JSON array: "[200000.0]"
    named_entities_count    INTEGER,
    jurisdiction_confidence FLOAT,

    PRIMARY KEY (regulation_id, version_id)
);

-- ------------------------------------------------------------
-- Jurisdictions reference table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS jurisdictions (
    jurisdiction_code   VARCHAR(10)     PRIMARY KEY,
    jurisdiction_name   VARCHAR(100)    NOT NULL,
    region              VARCHAR(50),
    gdpr_applicable     BOOLEAN         DEFAULT FALSE,
    notes               TEXT
);

INSERT INTO jurisdictions VALUES
    ('EU',      'European Union',   'Europe',        TRUE,  'GDPR applies; also DSA, DMA, AI Act'),
    ('UK',      'United Kingdom',   'Europe',        TRUE,  'UK GDPR post-Brexit; ICO enforcement'),
    ('US',      'United States',    'North America', FALSE, 'FTC, FCC, CFPB; state-level CCPA'),
    ('DE',      'Germany',          'Europe',        TRUE,  'BfDI; strong GDPR enforcement history'),
    ('FR',      'France',           'Europe',        TRUE,  'CNIL; active GDPR enforcement'),
    ('CA',      'Canada',           'North America', FALSE, 'PIPEDA; new Bill C-27'),
    ('AU',      'Australia',        'Asia-Pacific',  FALSE, 'Privacy Act 1988; OAIC'),
    ('UNKNOWN', 'Unknown',          NULL,            FALSE, 'Jurisdiction could not be determined')
ON CONFLICT (jurisdiction_code) DO NOTHING;

-- ------------------------------------------------------------
-- Pipeline health table
-- One row per connector run — powers the observability dashboard
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_health (
    run_id                      VARCHAR(24)     PRIMARY KEY,
    source                      VARCHAR(50)     NOT NULL,
    run_at                      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    rows_ingested               INTEGER         DEFAULT 0,
    duplicate_count             INTEGER         DEFAULT 0,
    null_rate                   FLOAT           DEFAULT 0.0,
    schema_drift                BOOLEAN         DEFAULT FALSE,
    sla_met                     BOOLEAN         DEFAULT TRUE,
    elapsed_seconds             FLOAT,
    errors                      TEXT,           -- JSON array
    quality_passed              BOOLEAN,
    quality_critical_failures   INTEGER,
    quality_warnings            INTEGER
);

-- ------------------------------------------------------------
-- Chunks table
-- One row per embeddable chunk — populated by vector_loader.py
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            VARCHAR(24)     PRIMARY KEY,
    regulation_id       VARCHAR         NOT NULL,
    version_id          INTEGER         NOT NULL DEFAULT 1,
    chunk_index         INTEGER         NOT NULL,
    total_chunks        INTEGER,
    text                TEXT            NOT NULL,
    token_estimate      INTEGER,
    jurisdiction        VARCHAR(10),
    regulation_type     VARCHAR(100),
    effective_date      DATE,
    source_url          VARCHAR(2000),
    page_number         INTEGER,
    article_ref         VARCHAR(100),
    section_ref         VARCHAR(100),
    chapter_ref         VARCHAR(100),
    block_type          VARCHAR(50),
    embedded_at         TIMESTAMPTZ,
    vector_store        VARCHAR(50),    -- "pinecone" | "chromadb"
    FOREIGN KEY (regulation_id, version_id)
        REFERENCES regulations(regulation_id, version_id)
);

-- ------------------------------------------------------------
-- Conflict signals table
-- Populated by conflict_detector.py
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conflict_signals (
    signal_id           VARCHAR(24)     PRIMARY KEY,
    jurisdiction_a      VARCHAR(10)     NOT NULL,
    jurisdiction_b      VARCHAR(10)     NOT NULL,
    regulation_id_a     VARCHAR         NOT NULL,
    regulation_id_b     VARCHAR         NOT NULL,
    topic               VARCHAR(200),
    conflict_summary    TEXT,
    severity            VARCHAR(20),    -- "high" | "medium" | "low"
    detected_at         TIMESTAMPTZ     DEFAULT NOW()
);

-- ============================================================
-- Views
-- ============================================================

-- Latest version of each regulation only
CREATE VIEW IF NOT EXISTS regulations_latest AS
SELECT r.*
FROM regulations r
INNER JOIN (
    SELECT regulation_id, MAX(version_id) AS max_version
    FROM regulations
    GROUP BY regulation_id
) latest
ON  r.regulation_id = latest.regulation_id
AND r.version_id    = latest.max_version;

-- Jurisdiction-level summary stats
CREATE VIEW IF NOT EXISTS jurisdiction_stats AS
SELECT
    r.jurisdiction,
    j.jurisdiction_name,
    COUNT(DISTINCT r.regulation_id)                         AS regulation_count,
    COUNT(DISTINCT r.regulation_type)                       AS type_diversity,
    MIN(r.effective_date)                                   AS earliest_date,
    MAX(r.effective_date)                                   AS latest_date,
    SUM(CASE WHEN r.version_id > 1 THEN 1 ELSE 0 END)      AS amendment_count
FROM regulations r
LEFT JOIN jurisdictions j ON r.jurisdiction = j.jurisdiction_code
GROUP BY r.jurisdiction, j.jurisdiction_name
ORDER BY regulation_count DESC;

-- Pipeline health last 7 days, grouped by source
CREATE VIEW IF NOT EXISTS pipeline_health_recent AS
SELECT
    source,
    COUNT(*)                                                AS total_runs,
    SUM(rows_ingested)                                      AS total_rows,
    ROUND(AVG(null_rate) * 100, 2)                          AS avg_null_rate_pct,
    SUM(CASE WHEN sla_met  = FALSE THEN 1 ELSE 0 END)       AS sla_breaches,
    SUM(CASE WHEN schema_drift = TRUE THEN 1 ELSE 0 END)    AS drift_events,
    MAX(run_at)                                             AS last_run_at
FROM pipeline_health
WHERE run_at >= NOW() - INTERVAL '7 days'
GROUP BY source
ORDER BY last_run_at DESC;

-- Amendment history — regulations that have more than one version
CREATE VIEW IF NOT EXISTS amendment_history AS
SELECT
    regulation_id,
    title,
    jurisdiction,
    version_id,
    effective_date,
    ingested_at,
    LAG(source_hash) OVER (
        PARTITION BY regulation_id ORDER BY version_id
    ) AS prev_hash,
    source_hash AS current_hash
FROM regulations
WHERE regulation_id IN (
    SELECT regulation_id
    FROM regulations
    GROUP BY regulation_id
    HAVING COUNT(*) > 1
)
ORDER BY regulation_id, version_id;