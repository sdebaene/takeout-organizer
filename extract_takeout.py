#!/usr/bin/env python3

import os
import sys
import tarfile
import shutil
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

def find_tgz_files(directory: Path) -> list[Path]:
    """Find all .tgz files in the given directory."""
    tgz_files = [f for f in directory.glob("*.tgz") if not f.name.startswith('.')]
    # Debug output
    print("\nDebug: Found TGZ files:")
    for tgz in sorted(tgz_files):
        print(f"  - {tgz.name}")
    return tgz_files

def merge_directories(src: Path, dest: Path) -> None:
    """Merge contents of source directory into destination directory."""
    if not dest.exists():
        shutil.move(str(src), str(dest))
        return

    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists() and target.is_dir():
                # Recursively merge directory contents
                merge_directories(item, target)
            else:
                # Move directory if no conflict
                shutil.move(str(item), str(target))
        else:
            if target.exists():
                # Check if files are identical (same size and modification time)
                src_stat = item.stat()
                target_stat = target.stat()
                
                if src_stat.st_size == target_stat.st_size and abs(src_stat.st_mtime - target_stat.st_mtime) < 1:
                    # Files are likely identical, skip
                    continue
                
                # Files are different or have different timestamps, handle conflict
                counter = 1
                new_name = item.name
                while (dest / new_name).exists():
                    stem = item.stem
                    suffix = item.suffix
                    new_name = f"{stem}_{counter}{suffix}"
                    counter += 1
                shutil.move(str(item), str(dest / new_name))
            else:
                # No conflict, just move the file
                shutil.move(str(item), str(target))

def process_tgz_file(tgz_file: Path, takeout_dir: Path) -> bool:
    """Process a single .tgz file."""
    # Create temporary directory next to the TGZ files
    temp_dir = tgz_file.parent / "tmp"
    temp_path = temp_dir / tgz_file.stem
    
    try:
        # Create temp directories
        temp_dir.mkdir(exist_ok=True)
        temp_path.mkdir(exist_ok=True)
        
        # Extract the archive
        with tarfile.open(tgz_file, 'r:gz') as tar:
            # Pre-filter members for security
            members_to_extract = []
            for member in tar.getmembers():
                # Skip unsafe paths
                if member.name.startswith('/') or '..' in member.name:
                    continue
                
                # Set safe permissions
                member.mode = 0o644 if member.isfile() else 0o755
                
                # Verify extraction path is safe
                target_path = os.path.join(str(temp_path), member.name)
                if os.path.commonprefix([str(temp_path), target_path]) == str(temp_path):
                    members_to_extract.append(member)
            
            # Extract filtered members
            tar.extractall(path=temp_path, members=members_to_extract)
        
        # Find and process Takeout directory
        takeout_source = next(temp_path.glob("Takeout"), None)
        if not takeout_source:
            print(f"No Takeout directory found in {tgz_file.name}")
            return False
        
        # Merge contents
        for item in takeout_source.iterdir():
            target = takeout_dir / item.name
            if item.is_dir():
                if target.exists():
                    merge_directories(item, target)
                else:
                    shutil.move(str(item), str(target))
            else:
                if target.exists():
                    # Handle file conflicts
                    counter = 1
                    new_name = item.name
                    while (takeout_dir / new_name).exists():
                        stem = item.stem
                        suffix = item.suffix
                        new_name = f"{stem}_{counter}{suffix}"
                        counter += 1
                    target = takeout_dir / new_name
                shutil.move(str(item), str(target))
        
        return True
        
    except Exception as e:
        print(f"Error processing {tgz_file.name}: {str(e)}")
        return False
        
    finally:
        # Clean up temporary directories
        if temp_path.exists():
            shutil.rmtree(temp_path)
        try:
            temp_dir.rmdir()  # Only removes if empty
        except OSError:
            pass  # Directory not empty is ok

def extract_takeout(directory: Path) -> None:
    """Extract and merge Takeout archives."""
    # Create necessary directories
    takeout_dir = directory / "Takeout"
    delete_dir = directory / "to_delete"
    
    takeout_dir.mkdir(exist_ok=True)
    delete_dir.mkdir(exist_ok=True)
    
    # Find all .tgz files
    tgz_files = find_tgz_files(directory)
    
    if not tgz_files:
        print("No .tgz files found")
        return
    
    print(f"Found {len(tgz_files)} .tgz files to process")
    
    # Determine number of processes to use (leave one CPU free for system)
    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} parallel processes")
    
    # Create a partial function with the takeout_dir parameter
    process_func = partial(process_tgz_file, takeout_dir=takeout_dir)
    
    # Process files in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_func, tgz_files)
    
    # Move processed files to delete directory
    success_count = 0
    for tgz_file, success in zip(tgz_files, results):
        if success:
            success_count += 1
            shutil.move(str(tgz_file), str(delete_dir / tgz_file.name))
            print(f"Successfully processed {tgz_file.name}")
        else:
            print(f"Failed to process {tgz_file.name}")
    
    print(f"""
Processing complete!
- Successfully processed: {success_count}/{len(tgz_files)} files
- Merged contents are in: {takeout_dir}
- Original .tgz files are in: {delete_dir}

Note: Please verify the contents of the Takeout directory before deleting the original files.""")

def main():
    parser = argparse.ArgumentParser(
        description='Extract and merge Google Takeout archives',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  Extract Takeout archives:
    %(prog)s /path/to/takeout/archives
        """
    )
    
    parser.add_argument('directory', help='Directory containing .tgz files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Convert to Path object and resolve to absolute path
    target_dir = Path(args.directory).resolve()
    
    # Check if directory exists
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Directory '{target_dir}' does not exist")
        sys.exit(1)
    
    # Execute extraction
    print(f"Extracting Takeout archives from: {target_dir}")
    extract_takeout(target_dir)

if __name__ == '__main__':
    main() 