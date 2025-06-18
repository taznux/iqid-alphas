#!/usr/bin/env python3
"""
ReUpload Dataset Structure Test

Quick test to explore the ReUpload dataset structure and verify our understanding
of the data organization before running full evaluation.
"""

import sys
from pathlib import Path

def explore_reupload_structure(reupload_path):
    """
    Explore the ReUpload dataset structure to understand organization
    """
    reupload_path = Path(reupload_path)
    print(f"🔍 Exploring ReUpload dataset: {reupload_path}")
    
    if not reupload_path.exists():
        print(f"❌ Path does not exist: {reupload_path}")
        return
    
    print(f"✅ Found ReUpload directory")
    
    # Look for the correct structure: ReUpload/iQID_reupload/iQID/3D/
    iqid_reupload_path = reupload_path / "iQID_reupload"
    if iqid_reupload_path.exists():
        print(f"📁 Found: iQID_reupload")
        
        iqid_path = iqid_reupload_path / "iQID"
        if iqid_path.exists():
            print(f"📁 Found: iQID")
            
            scans_3d_path = iqid_path / "3D"
            if scans_3d_path.exists():
                print(f"📁 Found: 3D")
                explore_3d_scans(scans_3d_path)
            else:
                print(f"❌ 3D directory not found in: {iqid_path}")
        else:
            print(f"❌ iQID directory not found in: {iqid_reupload_path}")
    else:
        print(f"❌ iQID_reupload directory not found in: {reupload_path}")
        
    # Also show what directories are present
    print(f"\n📋 All directories in ReUpload:")
    for item in reupload_path.iterdir():
        if item.is_dir():
            print(f"   📁 {item.name}")
        else:
            print(f"   📄 {item.name}")
    
def explore_3d_scans(scans_path):
    """
    Explore the 3D scans directory structure
    """
    print(f"   🔍 Exploring: {scans_path}")
    
    # Look for tissue type directories
    for tissue_dir in scans_path.iterdir():
        if tissue_dir.is_dir():
            print(f"   📂 Tissue type: {tissue_dir.name}")
            
            # Look for sample directories
            sample_count = 0
            for sample_dir in tissue_dir.iterdir():
                if sample_dir.is_dir():
                    sample_count += 1
                    if sample_count <= 3:  # Show first 3 samples
                        print(f"      📋 Sample: {sample_dir.name}")
                        explore_sample(sample_dir, indent="         ")
            
            if sample_count > 3:
                print(f"      ... and {sample_count - 3} more samples")

def explore_sample(sample_path, indent=""):
    """
    Explore a single sample directory
    """
    # Look for raw files
    raw_files = list(sample_path.glob("*_iqid_event_image.tif"))
    if raw_files:
        print(f"{indent}🎬 Raw files: {len(raw_files)}")
        for raw_file in raw_files[:2]:  # Show first 2
            print(f"{indent}   - {raw_file.name}")
    
    # Look for subdirectories
    subdirs = []
    for item in sample_path.iterdir():
        if item.is_dir():
            subdirs.append(item.name)
    
    if subdirs:
        print(f"{indent}📁 Subdirectories: {', '.join(subdirs)}")
        
        # Explore specific subdirectories
        for subdir_name in ['1_segmented', '2_aligned']:
            subdir = sample_path / subdir_name
            if subdir.exists():
                files = list(subdir.glob("*.tif"))
                print(f"{indent}   {subdir_name}: {len(files)} .tif files")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore ReUpload dataset structure")
    parser.add_argument("--data", required=True, help="Path to ReUpload dataset")
    
    args = parser.parse_args()
    
    explore_reupload_structure(args.data)

if __name__ == "__main__":
    main()
