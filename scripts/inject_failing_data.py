"""
Script to inject failing data into the Airbnb dbt project for demo purposes.

This script adds sample records that will cause test failures:
1. New sentiment values ('mixed', 'unknown') in reviews
2. NULL host_id in listings (migration scenario)
3. New room_type 'Studio' in listings
4. Out-of-range prices in listings
5. Orphan reviews (referencing non-existent listings)
"""

import csv
import os
from datetime import datetime, timedelta
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DBT_PROJECT = Path("/Users/duc.tran/airbnb-dbt-project")
SOURCE_DATA = DBT_PROJECT / "final_project_source_data"

# Backup original files
BACKUP_DIR = SOURCE_DATA / "backups"


def backup_file(filepath: Path):
    """Create a backup of a file."""
    BACKUP_DIR.mkdir(exist_ok=True)
    backup_path = BACKUP_DIR / f"{filepath.stem}_backup{filepath.suffix}"
    
    if not backup_path.exists():
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up {filepath.name} to {backup_path.name}")


def inject_failing_reviews():
    """Add reviews with new sentiment values."""
    reviews_path = SOURCE_DATA / "reviews.csv"
    backup_file(reviews_path)
    
    # Read existing reviews
    with open(reviews_path, 'r') as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fieldnames = reader.fieldnames
    
    # Get max listing_id from existing
    existing_listing_ids = set(r['listing_id'] for r in existing)
    sample_listing_id = list(existing_listing_ids)[0]
    
    # New failing records
    failing_reviews = [
        {
            'listing_id': sample_listing_id,
            'review_date': (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%dT00:00:00Z'),
            'reviewer_name': 'Alex Test',
            'comments': 'Mixed feelings - great location but noisy neighbors.',
            'sentiment': 'mixed'  # New value!
        },
        {
            'listing_id': sample_listing_id,
            'review_date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%dT00:00:00Z'),
            'reviewer_name': 'Jordan Demo',
            'comments': 'Could not analyze this review properly.',
            'sentiment': 'unknown'  # New value!
        },
        {
            'listing_id': sample_listing_id,
            'review_date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z'),
            'reviewer_name': 'Taylor Sample',
            'comments': 'Both good and bad aspects to this stay.',
            'sentiment': 'mixed'  # New value!
        },
        {
            'listing_id': '99999',  # Non-existent listing - orphan!
            'review_date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%dT00:00:00Z'),
            'reviewer_name': 'Orphan Review',
            'comments': 'This review has no matching listing.',
            'sentiment': 'positive'
        },
    ]
    
    # Write updated file
    with open(reviews_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(failing_reviews)
    
    print(f"✅ Added {len(failing_reviews)} failing reviews to {reviews_path.name}")


def inject_failing_listings():
    """Add listings with failing data."""
    listings_path = SOURCE_DATA / "listings.csv"
    backup_file(listings_path)
    
    # Read existing listings
    with open(listings_path, 'r') as f:
        reader = csv.DictReader(f)
        existing = list(reader)
        fieldnames = reader.fieldnames
    
    # Get max id
    max_id = max(int(r['id']) for r in existing)
    
    # New failing records
    failing_listings = [
        {
            'id': str(max_id + 1),
            'listing_url': f'https://www.airbnb.com/rooms/{max_id + 1}',
            'name': 'Modern City Studio',
            'room_type': 'Studio',  # New value!
            'minimum_nights': '2',
            'host_id': '12345',
            'price': '95.00',
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'updated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        {
            'id': str(max_id + 2),
            'listing_url': f'https://www.airbnb.com/rooms/{max_id + 2}',
            'name': 'Cozy Artist Studio',
            'room_type': 'Studio',  # New value!
            'minimum_nights': '1',
            'host_id': '12346',
            'price': '110.00',
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'updated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        {
            'id': str(max_id + 3),
            'listing_url': f'https://www.airbnb.com/rooms/{max_id + 3}',
            'name': 'Listing Under Migration',
            'room_type': 'Entire home/apt',
            'minimum_nights': '3',
            'host_id': '',  # NULL host_id!
            'price': '85.00',
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'updated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        {
            'id': str(max_id + 4),
            'listing_url': f'https://www.airbnb.com/rooms/{max_id + 4}',
            'name': 'First Night Free Promo',
            'room_type': 'Private room',
            'minimum_nights': '2',
            'host_id': '12347',
            'price': '0.00',  # Zero price!
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'updated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
        {
            'id': str(max_id + 5),
            'listing_url': f'https://www.airbnb.com/rooms/{max_id + 5}',
            'name': 'Ultra Luxury Penthouse',
            'room_type': 'Entire home/apt',
            'minimum_nights': '7',
            'host_id': '12348',
            'price': '15000.00',  # Above cap!
            'created_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
            'updated_at': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        },
    ]
    
    # Write updated file
    with open(listings_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(failing_listings)
    
    print(f"✅ Added {len(failing_listings)} failing listings to {listings_path.name}")


def restore_backups():
    """Restore original files from backups."""
    if not BACKUP_DIR.exists():
        print("❌ No backups found")
        return
    
    for backup in BACKUP_DIR.glob("*_backup.*"):
        original_name = backup.name.replace("_backup", "")
        original_path = SOURCE_DATA / original_name
        
        import shutil
        shutil.copy2(backup, original_path)
        print(f"✅ Restored {original_name}")


def main():
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        print("🔄 Restoring original files...")
        restore_backups()
        return
    
    print("🔧 Injecting failing data for demo...")
    print()
    
    inject_failing_reviews()
    inject_failing_listings()
    
    print()
    print("✅ Done! Run 'dbt test' to see the failures.")
    print("💡 To restore original files, run: python inject_failing_data.py --restore")


if __name__ == "__main__":
    main()

