#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Verification and Summary Report
Verify that data was correctly uploaded to Neo4j and generate extraction summary
"""

import sys
import os
from datetime import datetime

# Fix Windows Unicode
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import GraphDatabase


class Neo4jVerifier:
    """Verify and report on Neo4j data"""

    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="novelprocessing2024"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, params=None):
        """Execute Cypher query and return results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def verify_database_structure(self):
        """Verify Neo4j schema and indexes"""
        print("=" * 80)
        print("🔍 DATABASE STRUCTURE VERIFICATION")
        print("=" * 80)
        print()

        # Check node types
        print("📊 Node Types:")
        query = """
        MATCH (n)
        RETURN DISTINCT labels(n) as labels, count(*) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        for record in results:
            labels = record['labels'][0] if record['labels'] else 'No Label'
            count = record['count']
            print(f"   • {labels}: {count} nodes")
        print()

        # Check relationship types
        print("🔗 Relationship Types:")
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, count(*) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        for record in results:
            rel_type = record['rel_type']
            count = record['count']
            print(f"   • {rel_type}: {count} relationships")
        print()

        # Check indexes
        print("📇 Indexes:")
        query = "SHOW INDEXES"
        try:
            results = self.run_query(query)
            if results:
                for idx in results[:5]:  # Show first 5
                    print(f"   • {idx.get('name', 'unnamed')}: {idx.get('labelsOrTypes', 'N/A')}")
            else:
                print("   ⚠️  No indexes found")
        except Exception as e:
            print(f"   ⚠️  Could not query indexes: {e}")
        print()

    def verify_event_data(self):
        """Verify Event nodes"""
        print("=" * 80)
        print("📝 EVENT DATA VERIFICATION")
        print("=" * 80)
        print()

        # Total events by volume
        print("📊 Events by Volume:")
        query = """
        MATCH (e:Event)
        RETURN e.volume_id as volume, count(*) as event_count
        ORDER BY volume
        """
        results = self.run_query(query)
        total_events = 0
        for record in results:
            volume = record['volume']
            count = record['event_count']
            total_events += count
            print(f"   Volume {volume}: {count} events")
        print(f"   TOTAL: {total_events} events")
        print()

        # Sample events
        print("📋 Sample Events (first 5):")
        query = """
        MATCH (e:Event)
        RETURN e.event_id as id, e.volume_id as volume, e.description as desc,
               e.importance_score as importance, e.event_type as type
        ORDER BY e.volume_id, e.chronological_order
        LIMIT 5
        """
        results = self.run_query(query)
        for i, record in enumerate(results, 1):
            print(f"   {i}. [Vol {record['volume']}] {record['desc'][:60]}...")
            print(f"      Type: {record['type']}, Importance: {record['importance']:.2f}")
        print()

        # Events with missing data
        print("⚠️  Data Quality Check:")
        query = """
        MATCH (e:Event)
        WHERE e.description IS NULL OR e.description = ''
           OR e.event_type IS NULL
           OR e.importance_score IS NULL
        RETURN count(*) as incomplete_events
        """
        results = self.run_query(query)
        incomplete = results[0]['incomplete_events'] if results else 0
        if incomplete > 0:
            print(f"   ⚠️  {incomplete} events have incomplete data")
        else:
            print(f"   ✅ All events have complete required fields")
        print()

    def verify_causality_data(self):
        """Verify CAUSES relationships"""
        print("=" * 80)
        print("🔗 CAUSALITY DATA VERIFICATION")
        print("=" * 80)
        print()

        # Total causal links
        print("📊 Causal Links by Volume:")
        query = """
        MATCH (e1:Event)-[r:CAUSES]->(e2:Event)
        RETURN e1.volume_id as volume, count(r) as link_count
        ORDER BY volume
        """
        results = self.run_query(query)
        total_links = 0
        for record in results:
            volume = record['volume']
            count = record['link_count']
            total_links += count
            print(f"   Volume {volume}: {count} causal links")
        print(f"   TOTAL: {total_links} causal links")
        print()

        # Sample causal chains
        print("🔗 Sample Causal Chains:")
        query = """
        MATCH path = (e1:Event)-[:CAUSES]->(e2:Event)
        WHERE e1.volume_id = 1
        RETURN e1.description as from_event, e2.description as to_event,
               e1.event_id as from_id, e2.event_id as to_id
        LIMIT 5
        """
        results = self.run_query(query)
        for i, record in enumerate(results, 1):
            print(f"   {i}. {record['from_event'][:50]}...")
            print(f"      ➜ {record['to_event'][:50]}...")
        print()

        # Causal chain statistics
        print("📈 Causality Statistics:")

        # Events with no outgoing causes
        query = """
        MATCH (e:Event)
        WHERE NOT (e)-[:CAUSES]->()
        RETURN count(*) as terminal_events
        """
        results = self.run_query(query)
        terminal = results[0]['terminal_events'] if results else 0
        print(f"   Terminal events (no consequences): {terminal}")

        # Events with no incoming causes
        query = """
        MATCH (e:Event)
        WHERE NOT ()-[:CAUSES]->(e)
        RETURN count(*) as root_events
        """
        results = self.run_query(query)
        root = results[0]['root_events'] if results else 0
        print(f"   Root events (no causes): {root}")

        # Longest causal chain
        query = """
        MATCH p = (start:Event)-[:CAUSES*]->(end:Event)
        WHERE NOT ()-[:CAUSES]->(start)
        RETURN length(p) as chain_length
        ORDER BY chain_length DESC
        LIMIT 1
        """
        results = self.run_query(query)
        max_chain = results[0]['chain_length'] if results else 0
        print(f"   Longest causal chain: {max_chain} events")
        print()

    def verify_character_data(self):
        """Verify Character nodes"""
        print("=" * 80)
        print("🎭 CHARACTER DATA VERIFICATION")
        print("=" * 80)
        print()

        # Total characters by volume
        print("📊 Characters by Volume:")
        query = """
        MATCH (c:Character)
        RETURN c.volume_id as volume, count(*) as char_count
        ORDER BY volume
        """
        results = self.run_query(query)
        total_chars = 0
        for record in results:
            volume = record['volume']
            count = record['char_count']
            total_chars += count
            print(f"   Volume {volume}: {count} characters")
        print(f"   TOTAL: {total_chars} characters")
        print()

        # Characters by type
        print("📊 Characters by Type:")
        query = """
        MATCH (c:Character)
        RETURN c.character_type as type, count(*) as count
        ORDER BY count DESC
        """
        results = self.run_query(query)
        for record in results:
            char_type = record['type'] or 'Unknown'
            count = record['count']
            print(f"   • {char_type}: {count} characters")
        print()

        # Sample characters
        print("👥 Sample Characters (top 10 by appearance):")
        query = """
        MATCH (c:Character)
        RETURN c.name as name, c.character_type as type,
               c.volume_id as volume, c.confidence_score as confidence
        ORDER BY c.volume_id, confidence DESC
        LIMIT 10
        """
        results = self.run_query(query)
        for i, record in enumerate(results, 1):
            name = record['name']
            char_type = record['type'] or 'Unknown'
            volume = record['volume']
            confidence = record['confidence'] or 0.0
            print(f"   {i}. {name} ({char_type}) - Vol {volume} - Confidence: {confidence:.2f}")
        print()

    def verify_data_integrity(self):
        """Check data integrity and consistency"""
        print("=" * 80)
        print("✓ DATA INTEGRITY CHECKS")
        print("=" * 80)
        print()

        checks_passed = 0
        total_checks = 0

        # Check 1: All events have valid volume IDs
        total_checks += 1
        query = """
        MATCH (e:Event)
        WHERE e.volume_id IS NULL OR e.volume_id < 1
        RETURN count(*) as invalid_volumes
        """
        results = self.run_query(query)
        invalid = results[0]['invalid_volumes'] if results else 0
        if invalid == 0:
            print("   ✅ All events have valid volume IDs")
            checks_passed += 1
        else:
            print(f"   ❌ {invalid} events have invalid volume IDs")

        # Check 2: Causal links connect valid events
        total_checks += 1
        query = """
        MATCH (e1:Event)-[r:CAUSES]->(e2:Event)
        WHERE e1.event_id IS NULL OR e2.event_id IS NULL
        RETURN count(r) as invalid_links
        """
        results = self.run_query(query)
        invalid = results[0]['invalid_links'] if results else 0
        if invalid == 0:
            print("   ✅ All causal links connect valid events")
            checks_passed += 1
        else:
            print(f"   ❌ {invalid} causal links have invalid event references")

        # Check 3: No duplicate event IDs
        total_checks += 1
        query = """
        MATCH (e:Event)
        WITH e.event_id as event_id, count(*) as count
        WHERE count > 1
        RETURN sum(count) as duplicates
        """
        results = self.run_query(query)
        duplicates = results[0]['duplicates'] if results and results[0]['duplicates'] else 0
        if duplicates == 0:
            print("   ✅ No duplicate event IDs found")
            checks_passed += 1
        else:
            print(f"   ❌ {duplicates} duplicate event IDs found")

        # Check 4: Characters have names
        total_checks += 1
        query = """
        MATCH (c:Character)
        WHERE c.name IS NULL OR c.name = ''
        RETURN count(*) as nameless
        """
        results = self.run_query(query)
        nameless = results[0]['nameless'] if results else 0
        if nameless == 0:
            print("   ✅ All characters have names")
            checks_passed += 1
        else:
            print(f"   ❌ {nameless} characters are missing names")

        print()
        print(f"🎯 Integrity Score: {checks_passed}/{total_checks} checks passed")
        print()

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print()
        print("=" * 80)
        print("📊 COMPREHENSIVE SUMMARY REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print()

        # Overall statistics
        query = """
        MATCH (e:Event)
        WITH count(e) as total_events,
             count(DISTINCT e.volume_id) as volumes
        MATCH (c:Character)
        WITH total_events, volumes, count(c) as total_characters
        MATCH ()-[r:CAUSES]->()
        RETURN total_events, total_characters, count(r) as total_links, volumes
        """
        results = self.run_query(query)

        if results:
            stats = results[0]
            print("📈 OVERALL STATISTICS:")
            print(f"   Volumes Processed: {stats['volumes']}")
            print(f"   Total Events: {stats['total_events']}")
            print(f"   Total Characters: {stats['total_characters']}")
            print(f"   Total Causal Links: {stats['total_links']}")
            print()

            # Calculate derived metrics
            if stats['total_events'] > 0:
                avg_links_per_event = stats['total_links'] / stats['total_events']
                print(f"   Avg Causal Links per Event: {avg_links_per_event:.2f}")

            if stats['volumes'] > 0:
                avg_events_per_volume = stats['total_events'] / stats['volumes']
                avg_chars_per_volume = stats['total_characters'] / stats['volumes']
                print(f"   Avg Events per Volume: {avg_events_per_volume:.1f}")
                print(f"   Avg Characters per Volume: {avg_chars_per_volume:.1f}")
            print()

        # Top events by importance
        print("⭐ TOP 10 MOST IMPORTANT EVENTS:")
        query = """
        MATCH (e:Event)
        RETURN e.volume_id as volume, e.description as desc,
               e.importance_score as importance
        ORDER BY importance DESC
        LIMIT 10
        """
        results = self.run_query(query)
        for i, record in enumerate(results, 1):
            volume = record['volume']
            desc = record['desc'][:70] if record['desc'] else 'No description'
            importance = record['importance'] or 0.0
            print(f"   {i}. [Vol {volume}] {desc}...")
            print(f"      Importance: {importance:.2f}")
        print()

        print("=" * 80)
        print("✅ REPORT COMPLETE")
        print("=" * 80)


def main():
    """Main execution"""
    print()
    print("=" * 80)
    print("🔍 NEO4J DATABASE VERIFICATION & REPORT")
    print("=" * 80)
    print()

    try:
        verifier = Neo4jVerifier()

        # Run all verifications
        verifier.verify_database_structure()
        verifier.verify_event_data()
        verifier.verify_causality_data()
        verifier.verify_character_data()
        verifier.verify_data_integrity()
        verifier.generate_summary_report()

        verifier.close()

        print()
        print("💡 Next Steps:")
        print("   1. Open Neo4j Browser: http://localhost:7474")
        print("   2. Run custom queries to explore your data")
        print("   3. Visualize character networks and causality chains")
        print()

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
