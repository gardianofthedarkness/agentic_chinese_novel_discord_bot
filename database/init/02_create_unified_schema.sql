-- Unified Novel Processing Database Schema

-- Create unified results table (PostgreSQL version)
CREATE TABLE IF NOT EXISTS unified_results (
    id SERIAL PRIMARY KEY,
    volume_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    chunks_processed INTEGER NOT NULL,
    total_iterations INTEGER NOT NULL,
    final_satisfaction REAL NOT NULL,
    processing_stage TEXT NOT NULL,
    processing_mode TEXT NOT NULL,
    meaningful_improvements INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    early_terminated BOOLEAN DEFAULT FALSE,
    characters_detected TEXT,
    processing_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_unified_results_volume_id ON unified_results(volume_id);
CREATE INDEX IF NOT EXISTS idx_unified_results_created_at ON unified_results(created_at);
CREATE INDEX IF NOT EXISTS idx_unified_results_processing_stage ON unified_results(processing_stage);

-- Create unique constraint for volume_id + batch_id combination
ALTER TABLE unified_results ADD CONSTRAINT unique_volume_batch UNIQUE (volume_id, batch_id);

-- Create character tracking table
CREATE TABLE IF NOT EXISTS character_analysis (
    id SERIAL PRIMARY KEY,
    volume_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    character_name TEXT NOT NULL,
    character_role TEXT,
    key_actions TEXT[],
    appearance_frequency INTEGER DEFAULT 1,
    confidence_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Note: Foreign key constraint would require the referenced columns to have unique constraint
    -- FOREIGN KEY (volume_id, batch_id) REFERENCES unified_results(volume_id, batch_id) ON DELETE CASCADE,
    UNIQUE(volume_id, batch_id, character_name)
);

-- Create events tracking table
CREATE TABLE IF NOT EXISTS event_analysis (
    id SERIAL PRIMARY KEY,
    volume_id INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    event_description TEXT NOT NULL,
    importance_score REAL DEFAULT 0.0,
    timeline_position TEXT,
    event_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Note: Foreign key constraint would require the referenced columns to have unique constraint
    -- FOREIGN KEY (volume_id, batch_id) REFERENCES unified_results(volume_id, batch_id) ON DELETE CASCADE
);

-- Create character registry table for agentic character management
CREATE TABLE IF NOT EXISTS character_registry (
    id SERIAL PRIMARY KEY,
    volume_id INTEGER NOT NULL,
    character_name TEXT NOT NULL,
    character_role TEXT,
    character_description TEXT,
    confidence_score REAL DEFAULT 0.5,
    appearance_count INTEGER DEFAULT 1,
    first_appearance TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(volume_id, character_name)
);

-- Create indexes for character registry
CREATE INDEX IF NOT EXISTS idx_character_registry_volume_id ON character_registry(volume_id);
CREATE INDEX IF NOT EXISTS idx_character_registry_confidence ON character_registry(confidence_score DESC);

-- Migrate existing SQLite data (if needed)
-- This would be done by a separate migration script