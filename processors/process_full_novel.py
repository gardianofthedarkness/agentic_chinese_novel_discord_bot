#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full Novel Processing Script - Neo4j + LangGraph Integration
Process an entire novel volume through the complete pipeline
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime

# Fix Windows Unicode
if sys.platform == 'win32':
    import codecs
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_novel_processor import UnifiedNovelProcessor, ProcessingConfig, ProcessingMode


def print_banner():
    """Print startup banner"""
    print()
    print("=" * 100)
    print("🚀 NOVEL PROCESSING SYSTEM - Enhanced Logging Mode")
    print("=" * 100)
    print()




async def process_novel(volume_id: int, batch_size: int = 5, use_postgres: bool = False,
                       use_neo4j: bool = True, use_langgraph: bool = True,
                       enable_smart_routing: bool = True):
    """
    Process a full novel volume through the pipeline

    Args:
        volume_id: Volume ID to process
        batch_size: Number of chunks per batch
        use_postgres: Enable PostgreSQL backend
        use_neo4j: Enable Neo4j backend
        use_langgraph: Enable LangGraph workflow
        enable_smart_routing: Enable smart query routing
    """

    print_banner()

    # Configure processing
    config = ProcessingConfig(
        mode=ProcessingMode.BATCH,
        batch_size=batch_size,
        use_qdrant=True,  # Need Qdrant for data loading
        use_postgres=use_postgres,
        use_neo4j=use_neo4j,
        enable_smart_routing=enable_smart_routing,
        use_langgraph=use_langgraph,
        max_iterations=1  # Single pass for speed
    )

    print("=" * 100)
    print("📋 PROCESSING CONFIGURATION")
    print("=" * 100)
    print(f"  Volume ID: {volume_id}")
    print(f"  Batch size: {batch_size} chunks per batch")
    print(f"  Database backend: {'Neo4j' if use_neo4j and not use_postgres else 'Neo4j + PostgreSQL' if use_neo4j and use_postgres else 'PostgreSQL'}")
    print(f"  LangGraph workflow: {'Enabled' if use_langgraph else 'Disabled'}")
    print(f"  Smart routing: {'Enabled' if enable_smart_routing else 'Disabled'}")
    print("=" * 100)
    print()

    # Initialize processor
    print("🚀 Initializing processor...")
    processor = UnifiedNovelProcessor(config)
    print()

    # Statistics
    stats = {
        'batches_processed': 0,
        'total_characters': 0,
        'total_events': 0,
        'total_causal_links': 0,
        'total_errors': 0,
        'processing_time': 0,
        'start_time': datetime.now()
    }

    try:
        # Load data from Qdrant
        print(f"📖 Loading novel data for volume {volume_id}...")
        volume_chunks_dict = processor.data_loader.load_volume_chunks([volume_id])  # Pass as list

        # Extract chunks for this volume
        chunks = volume_chunks_dict.get(volume_id, [])

        if not chunks:
            print(f"❌ No chunks found for volume {volume_id}")
            print(f"   Check that volume {volume_id} exists in Qdrant collection '{config.collection_name}'")
            return

        print(f"   ✅ Loaded {len(chunks)} chunks")
        print()

        # Calculate batches
        num_batches = (len(chunks) + batch_size - 1) // batch_size

        print()
        print("#" * 100)
        print(f"📚 PROCESSING VOLUME {volume_id} - {num_batches} batches")
        print("#" * 100)
        print()

        # Track characters/events across batches for continuity
        previous_characters = {}
        previous_events = []

        volume_start_time = datetime.now()

        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            if use_langgraph and processor.langgraph_workflow:
                # Use LangGraph workflow
                result = await processor.langgraph_workflow.process_batch(
                    batch_id=batch_idx + 1,
                    volume_id=volume_id,
                    chunks=batch_chunks,
                    previous_characters=previous_characters,
                    previous_events=previous_events
                )

                # Update stats
                chars = result.get('characters', [])
                events = result.get('events', [])
                links = result.get('causal_links', [])
                errors = result.get('errors', [])

                if isinstance(chars, dict):
                    chars = list(chars.values())

                stats['total_characters'] += len(chars)
                stats['total_events'] += len(events)
                stats['total_causal_links'] += len(links)
                stats['total_errors'] += len(errors)

                # Update continuity context
                if isinstance(result.get('characters', []), list):
                    for char in result.get('characters', []):
                        char_id = char.get('character_id', char.get('name', f'char_{batch_idx}'))
                        previous_characters[char_id] = char

                previous_events.extend([evt.get('event_id') for evt in events if evt.get('event_id')])

            else:
                # Fallback: Use regular processor (no LangGraph)
                print()
                print("=" * 100)
                print(f"⚠️  BATCH {batch_idx + 1}/{num_batches} - LangGraph Not Available")
                print("=" * 100)
                print(f"   Using standard processing (not implemented)")
                print("=" * 100)
                print()

            stats['batches_processed'] += 1

        # Calculate volume processing time
        volume_end_time = datetime.now()
        volume_duration = (volume_end_time - volume_start_time).total_seconds()

        # Print volume completion summary
        print()
        print("=" * 100)
        print(f"✅ VOLUME {volume_id} COMPLETE")
        print("=" * 100)
        print(f"  Characters discovered: {stats['total_characters']}")
        print(f"  Events extracted: {stats['total_events']}")
        print(f"  Causal relationships: {stats['total_causal_links']}")
        print(f"  Processing time: {volume_duration/60:.1f} minutes")
        if stats['total_errors'] > 0:
            print(f"  ⚠️  Errors encountered: {stats['total_errors']}")
        print("=" * 100)
        print()

        # Calculate total processing time
        stats['processing_time'] = (datetime.now() - stats['start_time']).total_seconds()

        # Return stats for multi-volume tracking
        return stats

    except KeyboardInterrupt:
        print()
        print("=" * 100)
        print("⚠️  Processing interrupted by user")
        print("=" * 100)
        volume_duration = (datetime.now() - volume_start_time).total_seconds()
        print(f"  Processing time before interruption: {volume_duration/60:.1f} minutes")
        print(f"  Batches completed: {stats['batches_processed']}")
        print(f"  Characters: {stats['total_characters']}, Events: {stats['total_events']}, Links: {stats['total_causal_links']}")
        print("=" * 100)
        return stats

    except Exception as e:
        print()
        print("=" * 100)
        print(f"❌ Processing failed: {e}")
        print("=" * 100)
        import traceback
        traceback.print_exc()
        print("=" * 100)
        return None


def get_user_input():
    """Get processing parameters from user interactively"""
    print("=" * 80)
    print("📚 NOVEL PROCESSING - Interactive Setup")
    print("=" * 80)
    print()

    # Get volumes to process
    print("Which volumes would you like to process?")
    print()
    print("Options:")
    print("  1          - Process volume 1 only")
    print("  1 2 3      - Process volumes 1, 2, and 3")
    print("  1-5        - Process volumes 1 through 5")
    print("  all        - Process all available volumes")
    print()

    volume_input = input("Enter volume(s) to process: ").strip()

    # Parse volume input
    volumes = []
    if volume_input.lower() == 'all':
        # You can set a reasonable max here
        volumes = list(range(1, 21))  # Process volumes 1-20
        print(f"   → Will process volumes 1-20")
    elif '-' in volume_input:
        # Range format: 1-5
        try:
            start, end = volume_input.split('-')
            volumes = list(range(int(start.strip()), int(end.strip()) + 1))
            print(f"   → Will process volumes {volumes[0]}-{volumes[-1]}")
        except:
            print("   ❌ Invalid range format. Using volume 1.")
            volumes = [1]
    else:
        # Space-separated: 1 2 3
        try:
            volumes = [int(v.strip()) for v in volume_input.split()]
            print(f"   → Will process volumes: {', '.join(map(str, volumes))}")
        except:
            print("   ❌ Invalid input. Using volume 1.")
            volumes = [1]

    print()

    # Get batch size
    print("Batch size (chunks per batch):")
    print("  5  - Default (recommended)")
    print("  10 - Faster but more memory")
    print("  3  - Slower but more detailed")
    print()
    batch_input = input("Enter batch size [5]: ").strip()
    batch_size = int(batch_input) if batch_input else 5
    print(f"   → Using batch size: {batch_size}")
    print()

    # Get backend options
    print("Database backends:")
    print("  1 - Neo4j only (fastest, recommended)")
    print("  2 - Neo4j + PostgreSQL (hybrid, dual-write)")
    print("  3 - PostgreSQL only (legacy)")
    print()
    backend_input = input("Select backend [1]: ").strip()
    backend = int(backend_input) if backend_input else 1

    use_postgres = backend in [2, 3]
    use_neo4j = backend in [1, 2]

    print(f"   → PostgreSQL: {'✓' if use_postgres else '✗'}")
    print(f"   → Neo4j: {'✓' if use_neo4j else '✗'}")
    print()

    # LangGraph workflow
    print("Use LangGraph AI workflow? [Y/n]: ", end='')
    langgraph_input = input().strip().lower()
    use_langgraph = langgraph_input != 'n'
    print(f"   → LangGraph: {'✓' if use_langgraph else '✗'}")
    print()

    return {
        'volumes': volumes,
        'batch_size': batch_size,
        'use_postgres': use_postgres,
        'use_neo4j': use_neo4j,
        'use_langgraph': use_langgraph,
        'enable_smart_routing': use_neo4j  # Auto-enable if Neo4j is used
    }


async def process_multiple_volumes(volumes, **kwargs):
    """Process multiple volumes sequentially"""
    total_stats = {
        'volumes_processed': 0,
        'total_batches': 0,
        'total_characters': 0,
        'total_events': 0,
        'total_causal_links': 0,
        'total_errors': 0,
        'start_time': datetime.now()
    }

    for volume_idx, volume_id in enumerate(volumes, 1):
        print()
        print("#" * 100)
        print(f"📖 PROCESSING VOLUME {volume_id} ({volume_idx}/{len(volumes)})")
        print("#" * 100)
        print()

        try:
            volume_stats = await process_novel(volume_id=volume_id, **kwargs)
            if volume_stats:
                total_stats['volumes_processed'] += 1
                total_stats['total_batches'] += volume_stats.get('batches_processed', 0)
                total_stats['total_characters'] += volume_stats.get('total_characters', 0)
                total_stats['total_events'] += volume_stats.get('total_events', 0)
                total_stats['total_causal_links'] += volume_stats.get('total_causal_links', 0)
                total_stats['total_errors'] += volume_stats.get('total_errors', 0)
        except Exception as e:
            print()
            print("=" * 100)
            print(f"❌ VOLUME {volume_id} FAILED")
            print("=" * 100)
            print(f"  Error: {e}")
            print("=" * 100)
            print()
            continue

    # Final summary
    total_time = (datetime.now() - total_stats['start_time']).total_seconds()

    print()
    print("#" * 100)
    print("🎉 PROCESSING COMPLETE!")
    print("#" * 100)
    print()
    print("📊 FINAL STATISTICS")
    print("-" * 100)
    print(f"  Volumes processed: {total_stats['volumes_processed']}/{len(volumes)}")
    print(f"  Total batches: {total_stats['total_batches']}")
    print(f"  Total characters: {total_stats['total_characters']}")
    print(f"  Total events: {total_stats['total_events']}")
    print(f"  Total causal links: {total_stats['total_causal_links']}")
    print(f"  Total processing time: {total_time/60:.1f} minutes")
    if total_stats['volumes_processed'] > 0:
        print(f"  Average time per volume: {total_time/total_stats['volumes_processed']/60:.1f} minutes")
    if total_stats['total_errors'] > 0:
        print(f"  ⚠️  Total errors: {total_stats['total_errors']}")
    print("#" * 100)
    print()
    print("🔍 View results in Neo4j Browser: http://localhost:7474")
    print()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process novel volumes through Neo4j + LangGraph pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Interactive Mode (Default):
  python process_full_novel.py

  Then follow the prompts to select volumes and options.

Command-line Mode:
  python process_full_novel.py --volume 1
  python process_full_novel.py --volume 1 --use-postgres
  python process_full_novel.py --volume 1 --batch-size 10
        """
    )

    parser.add_argument('--volume', type=int,
                       help='Volume ID to process (omit for interactive mode)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Number of chunks per batch (default: 5)')
    parser.add_argument('--use-postgres', action='store_true',
                       help='Enable PostgreSQL backend')
    parser.add_argument('--no-neo4j', action='store_true',
                       help='Disable Neo4j backend')
    parser.add_argument('--no-langgraph', action='store_true',
                       help='Disable LangGraph workflow')
    parser.add_argument('--no-smart-routing', action='store_true',
                       help='Disable smart query routing')

    args = parser.parse_args()

    # Check if running in interactive mode
    if args.volume is None:
        # Interactive mode
        params = get_user_input()

        # Confirm before starting
        print("=" * 80)
        print("Ready to start processing!")
        print("Press Enter to continue, or Ctrl+C to cancel...")
        input()
        print()

        # Process volumes
        asyncio.run(process_multiple_volumes(**params))
    else:
        # Command-line mode
        asyncio.run(process_novel(
            volume_id=args.volume,
            batch_size=args.batch_size,
            use_postgres=args.use_postgres,
            use_neo4j=not args.no_neo4j,
            use_langgraph=not args.no_langgraph,
            enable_smart_routing=not args.no_smart_routing
        ))


if __name__ == "__main__":
    main()
