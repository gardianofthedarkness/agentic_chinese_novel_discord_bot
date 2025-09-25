#!/usr/bin/env python3
"""
Improved Unified System Runner
1. Starts Docker services (Qdrant + PostgreSQL)
2. SKIPS Volume 1 migration (already done)
3. CLEANS existing data for input volumes
4. Processes specified volumes with full continuity
"""

import os
import sys
import subprocess
import time
import asyncio
import psycopg2
from processors.unified_novel_processor import UnifiedNovelProcessor, ProcessingConfig, ProcessingMode

def check_docker():
    """Check if Docker is available"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Docker found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Docker not found. Please install Docker Desktop.")
    return False

def start_docker_services():
    """Start Docker services"""
    print("🐳 STARTING DOCKER SERVICES")
    print("=" * 60)
    
    if not check_docker():
        return False
    
    # Stop any existing services
    print("🛑 Stopping existing services...")
    subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'down'], 
                  capture_output=True)
    
    # Start services
    print("▶️  Starting unified services...")
    try:
        subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'up', '-d'], 
                      check=True)
        print("✅ Services started successfully!")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to initialize...")
        time.sleep(20)  # Give extra time for PostgreSQL to fully start
        
        # Check service status
        result = subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'ps'], 
                               capture_output=True, text=True)
        print("📊 Service Status:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False

def get_volume_input():
    """Get volume numbers from user input"""
    print(f"\n📋 VOLUME SELECTION:")
    print("Enter volumes to process (e.g., '2 3' or '1 5 7') or press Enter for volumes 2,3:")
    
    try:
        user_input = input("Volumes: ").strip()
        if user_input:
            volume_ids = [int(x) for x in user_input.split()]
        else:
            volume_ids = [2, 3]
    except ValueError:
        print("Invalid input, using default volumes 2,3")
        volume_ids = [2, 3]
    
    print(f"🎯 Selected volumes: {volume_ids}")
    return volume_ids

def clean_existing_data(volume_ids):
    """Clean existing data for specified volumes from PostgreSQL"""
    print(f"\n🧹 CLEANING EXISTING DATA FOR VOLUMES: {volume_ids}")
    print("=" * 60)
    
    try:
        # Connect to PostgreSQL
        print("🐘 Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            host='localhost',
            port=5433,
            database='novel_processing',
            user='novel_user',
            password='novel_pass'
        )
        cursor = conn.cursor()
        
        # Clean data for each volume
        total_deleted = 0
        for vol_id in volume_ids:
            # Check existing records
            cursor.execute("SELECT COUNT(*) FROM unified_results WHERE volume_id = %s", (vol_id,))
            existing_count = cursor.fetchone()[0]
            
            if existing_count > 0:
                print(f"   🗑️  Volume {vol_id}: Found {existing_count} existing records")
                
                # Delete existing records
                cursor.execute("DELETE FROM unified_results WHERE volume_id = %s", (vol_id,))
                deleted_count = cursor.rowcount
                total_deleted += deleted_count
                print(f"   ✅ Volume {vol_id}: Deleted {deleted_count} records")
            else:
                print(f"   ✨ Volume {vol_id}: No existing records (clean)")
        
        # Clean character analysis data
        for vol_id in volume_ids:
            cursor.execute("SELECT COUNT(*) FROM character_analysis WHERE volume_id = %s", (vol_id,))
            char_count = cursor.fetchone()[0]
            if char_count > 0:
                cursor.execute("DELETE FROM character_analysis WHERE volume_id = %s", (vol_id,))
                print(f"   🎭 Volume {vol_id}: Deleted {char_count} character records")
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ DATA CLEANING COMPLETE:")
        print(f"   📊 Total records deleted: {total_deleted}")
        print(f"   🎯 Volumes cleaned: {volume_ids}")
        print(f"   🚀 Ready for fresh processing!")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Data cleaning failed: {e}")
        print("🔄 Continuing with processing (data may contain previous test results)")
        return False

async def process_volumes(volume_ids):
    """Process specified volumes with unified system"""
    print(f"\n🚀 PROCESSING VOLUMES: {volume_ids}")
    print("=" * 60)
    
    # Configure for Docker environment
    config = ProcessingConfig(
        mode=ProcessingMode.ITERATIVE,
        max_iterations=3,
        satisfaction_threshold=0.80,
        batch_size=5,
        use_qdrant=True,
        qdrant_url="http://localhost:6333",
        collection_name="test_novel2",
        use_postgres=True,
        postgres_host="localhost",
        postgres_port=5433,
        postgres_db="novel_processing",
        postgres_user="novel_user",
        postgres_password="novel_pass"
    )
    
    print(f"🎯 Configuration:")
    print(f"   📊 Mode: {config.mode.value}")
    print(f"   🔗 Qdrant: {config.qdrant_url}")
    print(f"   🗄️  PostgreSQL: {config.postgres_host}:{config.postgres_port}")
    print(f"   📦 Batch size: {config.batch_size}")
    print(f"   📈 Max iterations: {config.max_iterations}")
    print(f"   🎯 Satisfaction threshold: {config.satisfaction_threshold}")
    
    # Process volumes
    processor = UnifiedNovelProcessor(config)
    result = await processor.process_volumes(volume_ids)
    
    return result

def display_final_summary(processing_result: dict, volume_ids: list):
    """Display comprehensive final summary"""
    print(f"\n" + "=" * 80)
    print("🎉 UNIFIED SYSTEM EXECUTION COMPLETE")
    print("=" * 80)
    
    if processing_result and processing_result.get('success'):
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"   Volumes: {processing_result['volumes_processed']}")
        print(f"   Total batches: {processing_result['total_batches']}")
        print(f"   Total chunks: {processing_result['total_chunks']}")
        print(f"   Avg satisfaction: {processing_result['avg_satisfaction']:.3f}")
        print(f"   Total cost: ${processing_result['total_cost']:.4f}")
        print(f"   Processing time: {processing_result['processing_time']:.1f}s ({processing_result['processing_time']/60:.1f}min)")
        
        print(f"\n🗄️ DATABASE STATUS:")
        print(f"   Volume 1: ✅ Previously migrated (preserved)")
        print(f"   Volumes {volume_ids}: ✅ Freshly processed (cleaned before run)")
        print(f"   Database: PostgreSQL (containerized)")
        
        if processing_result.get('volume_breakdown'):
            print(f"\n📖 VOLUME BREAKDOWN:")
            for vol_id, stats in processing_result['volume_breakdown'].items():
                print(f"   Volume {vol_id}: {stats['batches']} batches, {stats['chunks']} chunks, ${stats['cost']:.3f}")
    
    print(f"\n🐳 DOCKER SERVICES:")
    print(f"   Qdrant: http://localhost:6333")
    print(f"   PostgreSQL: localhost:5433")
    print(f"   To stop: docker-compose -f docker-compose-unified.yml down")
    
    print(f"\n📁 DATA LOCATIONS:")
    print(f"   PostgreSQL: Containerized volume (persistent)")
    print(f"   Qdrant: ./qdrant_data/")
    print(f"   Reports: ./unified_processing_report_*.json")

async def main():
    """Main unified system execution"""
    print("🚀 IMPROVED UNIFIED NOVEL PROCESSING SYSTEM")
    print("=" * 80)
    print("🎯 Workflow: Docker + Clean Data + Process Volumes")
    print("📊 Features: Skip migration + Clean previous test data")
    print("🗄️ Database: PostgreSQL + Qdrant")
    print("=" * 80)
    
    processing_result = None
    volume_ids = []
    
    try:
        # Step 1: Start Docker services
        if not start_docker_services():
            print("❌ Cannot continue without Docker services")
            return False
        
        # Step 2: Get volume input
        volume_ids = get_volume_input()
        
        # Step 3: Clean existing data for selected volumes
        clean_existing_data(volume_ids)
        
        # Step 4: Process selected volumes
        processing_result = await process_volumes(volume_ids)
        
        # Step 5: Display results
        display_final_summary(processing_result, volume_ids)
        
        if processing_result and processing_result.get('success'):
            print(f"\n🎉 SUCCESS! Clean unified processing completed for volumes {volume_ids}")
            return True
        else:
            print(f"\n⚠️ Processing had issues, but system is operational")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⏹️ System interrupted by user")
        return False
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)