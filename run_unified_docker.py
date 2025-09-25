#!/usr/bin/env python3
"""
Unified Docker Deployment Script
Starts all services: Backend, Qdrant, PostgreSQL
Automatically detects environment and configures services
"""

import os
import sys
import subprocess
import time
import asyncio
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

def start_unified_services():
    """Start unified Docker services"""
    print("🚀 Starting Unified Novel Processing Services...")
    print("=" * 60)
    
    if not check_docker():
        return False
    
    # Stop any existing services
    print("🛑 Stopping existing services...")
    subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'down'], 
                  capture_output=True)
    
    # Start services
    print("▶️  Starting services...")
    try:
        subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'up', '-d'], 
                      check=True)
        print("✅ Services started successfully!")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to be ready...")
        time.sleep(15)
        
        # Check service status
        result = subprocess.run(['docker-compose', '-f', 'docker-compose-unified.yml', 'ps'], 
                               capture_output=True, text=True)
        print("📊 Service Status:")
        print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start services: {e}")
        return False

async def run_processing_with_docker():
    """Run processing using Docker services"""
    print("\n🔧 CONFIGURING FOR DOCKER ENVIRONMENT")
    print("=" * 60)
    
    # Configure for Docker environment
    config = ProcessingConfig(
        mode=ProcessingMode.ITERATIVE,
        max_iterations=3,
        satisfaction_threshold=0.80,
        batch_size=5,
        use_qdrant=True,
        qdrant_url="http://localhost:6333",  # Docker exposed port
        collection_name="test_novel2",
        use_postgres=True,
        postgres_host="localhost",  # Docker exposed port
        postgres_port=5432,
        postgres_db="novel_processing",
        postgres_user="novel_user",
        postgres_password="novel_pass"
    )
    
    print(f"🎯 Configuration:")
    print(f"   📊 Mode: {config.mode.value}")
    print(f"   🔗 Qdrant: {config.qdrant_url}")
    print(f"   🗄️  PostgreSQL: {config.postgres_host}:{config.postgres_port}")
    print(f"   📦 Batch size: {config.batch_size}")
    
    # Get volume selection
    print("\n📋 Volume Selection:")
    print("Enter volume numbers to process (e.g., '2 3' for volumes 2 and 3)")
    print("Or just press Enter to process volumes 2 and 3 by default")
    
    user_input = input("Volumes to process: ").strip()
    
    if user_input:
        try:
            volume_ids = [int(x) for x in user_input.split()]
        except ValueError:
            print("❌ Invalid input. Using default volumes 2 and 3.")
            volume_ids = [2, 3]
    else:
        volume_ids = [2, 3]
    
    print(f"🎯 Selected volumes: {volume_ids}")
    
    # Create processor and run
    processor = UnifiedNovelProcessor(config)
    result = await processor.process_volumes(volume_ids)
    
    return result

def migrate_existing_data():
    """Migrate existing SQLite data to PostgreSQL"""
    print("\n📊 DATA MIGRATION OPTIONS")
    print("=" * 60)
    print("Found existing Volume 1 databases:")
    
    db_files = [
        'volume_1_enhanced_batch.db',
        'enhanced_iterative_results.db',
        'fixed_iterative_results.db',
        'limitless_1_volume_results.db'
    ]
    
    existing_dbs = []
    for db_file in db_files:
        if os.path.exists(db_file):
            existing_dbs.append(db_file)
            print(f"   📁 {db_file}")
    
    if existing_dbs:
        print(f"\n💡 Found {len(existing_dbs)} existing database(s)")
        print("These contain your Volume 1 processing results.")
        print("The unified system will create a new PostgreSQL database.")
        print("Your existing data will remain as backup.")
    
    return existing_dbs

async def main():
    """Main execution"""
    print("🚀 UNIFIED NOVEL PROCESSING - DOCKER DEPLOYMENT")
    print("=" * 80)
    print("🐳 Full containerized stack: Backend + Qdrant + PostgreSQL")
    print("📊 Unified processing with persistent data storage")
    print("=" * 80)
    
    try:
        # Check existing data
        migrate_existing_data()
        
        # Start Docker services
        if not start_unified_services():
            print("❌ Failed to start Docker services")
            return False
        
        # Run processing
        print(f"\n🚀 RUNNING UNIFIED PROCESSING")
        print("=" * 60)
        
        result = await run_processing_with_docker()
        
        if result and result.get('success'):
            print(f"\n🎉 SUCCESS!")
            print(f"📊 Processed {len(result['volumes_processed'])} volumes")
            print(f"💰 Total cost: ${result['total_cost']:.4f}")
            print(f"⏱️  Processing time: {result['processing_time']:.1f}s")
            print(f"\n🐳 Services remain running for future processing")
            print(f"🛑 To stop services: docker-compose -f docker-compose-unified.yml down")
            return True
        else:
            print("❌ Processing failed")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)