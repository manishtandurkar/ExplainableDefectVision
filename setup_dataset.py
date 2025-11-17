#!/usr/bin/env python3
"""
NEU Surface Defect Dataset Download Helper

This script helps download and organize the NEU Surface Defect Dataset
for the Explainable Defect Detection project.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

def setup_neu_dataset():
    """Setup NEU dataset from manual download"""
    
    print("🏭 NEU Surface Defect Dataset Setup Helper")
    print("=" * 50)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Look for downloaded ZIP file
    zip_files = list(data_dir.glob("*.zip"))
    
    if not zip_files:
        print("📥 Dataset ZIP file not found!")
        print("\n📝 Manual Download Instructions:")
        print("1. 🌐 Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
        print("2. 📁 Click 'Download' button")
        print("3. 📂 Move the downloaded ZIP file to the 'data/' folder")
        print("4. 🔄 Run this script again")
        return False
    
    # Use the first ZIP file found
    zip_file = zip_files[0]
    print(f"📦 Found dataset ZIP: {zip_file.name}")
    
    # Extract ZIP file
    print("📂 Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Find extracted folder
    extracted_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name != 'neu_defect_dataset']
    
    if not extracted_folders:
        print("❌ Could not find extracted dataset folder")
        return False
    
    source_dir = extracted_folders[0]
    target_dir = data_dir / "neu_defect_dataset"
    target_dir.mkdir(exist_ok=True)
    
    # Class mapping from NEU format to our format
    class_mapping = {
        'Cr': 'crazing',
        'In': 'inclusion', 
        'Pa': 'patches',
        'PS': 'pitted_surface',
        'RS': 'rolled_in_scale',
        'Sc': 'scratches'
    }
    
    print("🗂️ Organizing dataset structure...")
    
    # Organize files by class
    total_files = 0
    for orig_class, new_class in class_mapping.items():
        class_dir = target_dir / new_class
        class_dir.mkdir(exist_ok=True)
        
        # Find files for this class
        pattern_files = list(source_dir.rglob(f"*{orig_class}_*"))
        if not pattern_files:
            # Try alternative patterns
            pattern_files = list(source_dir.rglob(f"*{orig_class}*"))
        
        print(f"📋 Processing {orig_class} -> {new_class}: {len(pattern_files)} files")
        
        for i, file_path in enumerate(pattern_files):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                new_name = f"{new_class}_{i:03d}{file_path.suffix}"
                shutil.copy2(file_path, class_dir / new_name)
                total_files += 1
    
    print(f"✅ Dataset organized successfully!")
    print(f"📊 Total files processed: {total_files}")
    print(f"📁 Dataset location: {target_dir.absolute()}")
    
    # Clean up extracted folder if different from target
    if source_dir != target_dir:
        shutil.rmtree(source_dir)
    
    # Remove ZIP file
    os.remove(zip_file)
    
    return True

if __name__ == "__main__":
    if setup_neu_dataset():
        print("\n🚀 You can now run the notebook!")
        print("📓 Open: ExplainableDefectDetection.ipynb")
    else:
        print("\n❌ Setup failed. Please follow manual instructions.")