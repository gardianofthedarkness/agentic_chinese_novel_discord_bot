-- PostgreSQL initialization script for timeline system
-- This script runs automatically when PostgreSQL container starts

-- Create timeline nodes table with JSONB for flexible schema
CREATE TABLE IF NOT EXISTS timeline_nodes (
    id TEXT PRIMARY KEY,
    parent_id TEXT,
    channel_id TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL,
    story_timestamp TIMESTAMP,
    
    -- Store all timeline data as JSONB for flexibility
    data JSONB NOT NULL,
    
    -- Extracted fields for indexing and queries
    universe_hash TEXT,
    current_location TEXT,
    story_phase TEXT,
    
    -- Metadata
    is_milestone BOOLEAN DEFAULT FALSE,
    message_count INTEGER DEFAULT 0,
    
    FOREIGN KEY (parent_id) REFERENCES timeline_nodes(id)
);

-- Create character states table
CREATE TABLE IF NOT EXISTS character_states (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    character_name TEXT NOT NULL,
    character_type TEXT NOT NULL,
    user_id TEXT,
    
    -- Store character data as JSONB
    data JSONB NOT NULL,
    
    -- Extracted fields for queries
    current_location TEXT,
    emotional_state TEXT DEFAULT 'neutral',
    plot_importance REAL DEFAULT 0.5,
    
    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id) ON DELETE CASCADE,
    UNIQUE(node_id, character_name)
);

-- Create story events table
CREATE TABLE IF NOT EXISTS story_events (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    summary TEXT NOT NULL,
    impact_score REAL DEFAULT 1.0,
    
    -- Store event data as JSONB
    data JSONB NOT NULL,
    
    FOREIGN KEY (node_id) REFERENCES timeline_nodes(id) ON DELETE CASCADE
);

-- Create user characters table
CREATE TABLE IF NOT EXISTS user_characters (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    character_name TEXT NOT NULL,
    
    -- Store character data as JSONB
    data JSONB NOT NULL,
    
    -- Extracted fields
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    
    UNIQUE(user_id, character_name)
);

-- Create timeline branches table for easier management
CREATE TABLE IF NOT EXISTS timeline_branches (
    id TEXT PRIMARY KEY,
    channel_id TEXT NOT NULL,
    root_node_id TEXT NOT NULL,
    current_node_id TEXT NOT NULL,
    branch_name TEXT,
    created_at TIMESTAMP NOT NULL,
    last_active TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    FOREIGN KEY (root_node_id) REFERENCES timeline_nodes(id),
    FOREIGN KEY (current_node_id) REFERENCES timeline_nodes(id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_nodes_channel ON timeline_nodes(channel_id);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON timeline_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_created ON timeline_nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_universe ON timeline_nodes(universe_hash);
CREATE INDEX IF NOT EXISTS idx_nodes_location ON timeline_nodes(current_location);
CREATE INDEX IF NOT EXISTS idx_nodes_milestone ON timeline_nodes(is_milestone);

-- JSONB indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_nodes_data_universe ON timeline_nodes USING GIN ((data->'universe_fingerprint'));
CREATE INDEX IF NOT EXISTS idx_nodes_data_characters ON timeline_nodes USING GIN ((data->'character_states'));
CREATE INDEX IF NOT EXISTS idx_nodes_data_chapter ON timeline_nodes USING GIN ((data->'chapter_context'));

-- Character state indexes
CREATE INDEX IF NOT EXISTS idx_characters_node ON character_states(node_id);
CREATE INDEX IF NOT EXISTS idx_characters_name ON character_states(character_name);
CREATE INDEX IF NOT EXISTS idx_characters_user ON character_states(user_id);
CREATE INDEX IF NOT EXISTS idx_characters_type ON character_states(character_type);
CREATE INDEX IF NOT EXISTS idx_characters_location ON character_states(current_location);

-- Story event indexes
CREATE INDEX IF NOT EXISTS idx_events_node ON story_events(node_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON story_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON story_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_impact ON story_events(impact_score);

-- User character indexes
CREATE INDEX IF NOT EXISTS idx_userchars_user ON user_characters(user_id);
CREATE INDEX IF NOT EXISTS idx_userchars_active ON user_characters(is_active);
CREATE INDEX IF NOT EXISTS idx_userchars_created ON user_characters(created_at);

-- Timeline branch indexes
CREATE INDEX IF NOT EXISTS idx_branches_channel ON timeline_branches(channel_id);
CREATE INDEX IF NOT EXISTS idx_branches_active ON timeline_branches(is_active);
CREATE INDEX IF NOT EXISTS idx_branches_current ON timeline_branches(current_node_id);

-- Create functions for timeline management
CREATE OR REPLACE FUNCTION get_timeline_path(start_node_id TEXT)
RETURNS TABLE(node_id TEXT, depth INTEGER) AS $$
BEGIN
    RETURN QUERY
    WITH RECURSIVE timeline_path AS (
        -- Base case: start node
        SELECT t.id as node_id, 0 as depth
        FROM timeline_nodes t
        WHERE t.id = start_node_id
        
        UNION ALL
        
        -- Recursive case: parent nodes
        SELECT t.id as node_id, p.depth + 1 as depth
        FROM timeline_nodes t
        JOIN timeline_path p ON t.id = p.node_id
        WHERE t.parent_id IS NOT NULL
    )
    SELECT tp.node_id, tp.depth FROM timeline_path tp
    ORDER BY tp.depth;
END;
$$ LANGUAGE plpgsql;

-- Create function for similarity scoring
CREATE OR REPLACE FUNCTION calculate_universe_similarity(
    universe1 JSONB,
    universe2 JSONB
) RETURNS REAL AS $$
DECLARE
    char_similarity REAL := 0;
    location_similarity REAL := 0;
    plot_similarity REAL := 0;
    total_score REAL := 0;
BEGIN
    -- Character similarity (simplified)
    IF universe1->>'main_characters' = universe2->>'main_characters' THEN
        char_similarity := 0.3;
    END IF;
    
    -- Location similarity
    IF universe1->>'current_location' = universe2->>'current_location' THEN
        location_similarity := 0.2;
    END IF;
    
    -- Plot similarity (simplified)
    IF universe1->>'story_phase' = universe2->>'story_phase' THEN
        plot_similarity := 0.15;
    END IF;
    
    total_score := char_similarity + location_similarity + plot_similarity;
    
    RETURN total_score;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing (optional)
-- This can be removed in production
INSERT INTO timeline_nodes (id, parent_id, channel_id, created_at, data, universe_hash, current_location, story_phase)
VALUES (
    'sample_root_001',
    NULL,
    'sample_channel',
    NOW(),
    '{"chapter_context": {"title": "Sample Story", "setting": "Library"}, "universe_fingerprint": {"story_phase": "beginning", "current_location": "Library"}}',
    'sample_hash_001',
    'Library',
    'beginning'
) ON CONFLICT (id) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO timeline_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO timeline_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO timeline_user;