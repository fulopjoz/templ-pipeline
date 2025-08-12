#!/usr/bin/env python3
"""
Script to download and set up PDBBind data for TEMPL Pipeline
Uses the specific URLs and extraction process provided by the user
"""

import os
import sys
import requests
import tarfile
import shutil
from pathlib import Path

def download_file(url, target_dir="/app/data/PDBBind"):
    """Download file from URL with proper filename detection"""
    print(f"Downloading from: {url}")
    
    try:
        r = requests.get(url, allow_redirects=True, stream=True)
        r.raise_for_status()
        
        # Try to get filename from Content-Disposition header
        cd = r.headers.get('Content-Disposition', '')
        if 'filename=' in cd:
            filename = cd.split('filename=')[1].strip('"').strip("'")
        else:
            # Fallback filename based on URL pattern
            if 'yTIjU12mhtRsNmZ' in url:
                filename = 'PDBbind_v2020_refined.tar.gz'
            elif 'WUpmz163j45YrpF' in url:
                filename = 'PDBbind_v2020_other_PL.tar.gz'
            else:
                filename = 'downloaded_file.tar.gz'
        
        filepath = os.path.join(target_dir, filename)
        
        # Ensure target directory exists
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"Saving to: {filepath}")
        total_size = int(r.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownloaded: {filename} ({downloaded:,} bytes)")
        return filepath, filename
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None, None

def extract_tarfile(filepath, target_dir):
    """Extract tar.gz file to target directory"""
    print(f"Extracting {filepath} to {target_dir}")
    
    try:
        os.makedirs(target_dir, exist_ok=True)
        
        with tarfile.open(filepath, 'r:gz') as tar:
            # Get list of members for progress tracking
            members = tar.getmembers()
            total_members = len(members)
            
            for i, member in enumerate(members):
                tar.extract(member, target_dir)
                if (i + 1) % 100 == 0 or i + 1 == total_members:
                    percent = ((i + 1) / total_members) * 100
                    print(f"\rExtracting: {percent:.1f}% ({i+1}/{total_members})", end='', flush=True)
        
        print(f"\nExtracted {total_members} files to {target_dir}")
        return True
        
    except Exception as e:
        print(f"Error extracting {filepath}: {e}")
        return False

def setup_pdbind_data():
    """Main function to download and set up PDBBind data"""
    
    # URLs provided by user
    urls = [
        'https://owncloud.cesnet.cz/index.php/s/yTIjU12mhtRsNmZ/download',  # refined
        'https://owncloud.cesnet.cz/index.php/s/WUpmz163j45YrpF/download'   # other_PL
    ]
    
    base_dir = "/app/data/PDBBind"
    print(f"Setting up PDBBind data in: {base_dir}")
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    downloaded_files = []
    
    # Download both files
    for url in urls:
        filepath, filename = download_file(url, base_dir)
        if filepath and filename:
            downloaded_files.append((filepath, filename))
        else:
            print(f"Failed to download from {url}")
            return False
    
    print(f"\nDownloaded {len(downloaded_files)} files")
    
    # Extract files
    for filepath, filename in downloaded_files:
        if 'refined' in filename:
            target_dir = os.path.join(base_dir, 'PDBbind_v2020_refined')
            print(f"\nExtracting refined dataset...")
        elif 'other_PL' in filename:
            target_dir = os.path.join(base_dir, 'PDBbind_v2020_other_PL')
            print(f"\nExtracting other_PL dataset...")
        else:
            target_dir = os.path.join(base_dir, 'extracted')
            print(f"\nExtracting to generic directory...")
        
        success = extract_tarfile(filepath, target_dir)
        if not success:
            print(f"Failed to extract {filepath}")
            return False
        
        # Clean up downloaded tar.gz file
        try:
            os.remove(filepath)
            print(f"Removed downloaded file: {filepath}")
        except Exception as e:
            print(f"Could not remove {filepath}: {e}")
    
    # Verify the setup
    print("\n" + "="*50)
    print("PDBBind Data Setup Complete!")
    print("="*50)
    
    # List what was created
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            file_count = len(list(Path(item_path).rglob('*')))
            print(f"📁 {item}/: {file_count} files/directories")
        else:
            size = os.path.getsize(item_path)
            print(f"📄 {item}: {size:,} bytes")
    
    return True

if __name__ == "__main__":
    print("TEMPL Pipeline - PDBBind Data Setup")
    print("="*40)
    
    # Check if we're in the right environment
    if not os.path.exists("/app/data"):
        print("Error: /app/data directory not found")
        print("This script should be run inside the TEMPL pipeline container")
        sys.exit(1)
    
    success = setup_pdbind_data()
    if success:
        print("\n✅ PDBBind data setup completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ PDBBind data setup failed!")
        sys.exit(1)