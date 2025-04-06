#!/usr/bin/env python3

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Set
import json
from PIL import Image, UnidentifiedImageError
import piexif
import ffmpeg
from pillow_heif import register_heif_opener
from datetime import datetime, timezone
import subprocess
import shutil
import tempfile
import os
import multiprocessing
from functools import partial

# Register HEIF opener with Pillow
register_heif_opener()

# File extensions from the original script
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    '.avif', '.bmp', '.gif', '.heic', '.heif', '.ico', '.jp2',
    '.jpg', '.jpeg', '.jpe', '.insp', '.jxl', '.png', '.psd',
    '.raw', '.rw2', '.svg', '.tif', '.tiff', '.webp', '.crw',
    '.cr2', '.nef', '.dng'
}

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    '.3gp', '.3gpp', '.3g2', '.asf', '.avi', '.divx', '.flv',
    '.m4v', '.mkv', '.mmv', '.mod', '.mts', '.m2ts', '.m2t',
    '.mp4', '.insv', '.mpg', '.mpe', '.mpeg', '.mov', '.tod',
    '.webm', '.wmv', '.vob', '.lrv'
}

SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

# Define legacy video formats that should be converted to MP4
LEGACY_VIDEO_FORMATS: Set[str] = {'.avi', '.mpg', '.mpeg', '.mpe'}

class MetadataHandler:
    """Handles metadata operations for media files."""
    
    def __init__(self, file_path: Path, metadata: dict):
        self.file_path = file_path
        self.metadata = metadata
        self.temp_path = file_path.with_suffix('.temp' + file_path.suffix)
    
    def _get_datetime(self) -> tuple[str, Exception]:
        """Extract datetime from metadata."""
        try:
            if 'photoTakenTime' not in self.metadata:
                return None, None
                
            timestamp = self.metadata['photoTakenTime'].get('timestamp', '')
            if not timestamp:
                return None, None
                
            dt = parse_timestamp(timestamp)
            return dt, None
        except Exception as e:
            return None, e
    
    def _get_gps_data(self) -> tuple[dict, Exception]:
        """Extract GPS data from metadata."""
        try:
            if 'geoData' not in self.metadata:
                return None, None
                
            geo = self.metadata['geoData']
            if 'latitude' not in geo or 'longitude' not in geo:
                return None, None
                
            gps_data = {
                'latitude': float(geo['latitude']),
                'longitude': float(geo['longitude']),
                'altitude': float(geo.get('altitude', 0))
            }
            return gps_data, None
        except Exception as e:
            return None, e
    
    def _get_camera_info(self) -> tuple[dict, Exception]:
        """Extract camera information from metadata."""
        try:
            camera_info = {}
            if 'cameraMake' in self.metadata:
                camera_info['make'] = self.metadata['cameraMake']
            if 'cameraModel' in self.metadata:
                camera_info['model'] = self.metadata['cameraModel']
            return camera_info if camera_info else None, None
        except Exception as e:
            return None, e
    
    def _cleanup_temp_file(self):
        """Clean up temporary file if it exists."""
        if self.temp_path.exists():
            try:
                self.temp_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file {self.temp_path}: {e}")

    def _handle_error(self, error: Exception, context: str):
        """Handle errors consistently."""
        print(f"Error {context} for {self.file_path}: {error}")
        self._cleanup_temp_file()
        return False

    def _update_with_exiftool(self) -> bool:
        """Update metadata using ExifTool."""
        if not shutil.which('exiftool'):
            print(f"Error: ExifTool is not installed. Cannot process file {self.file_path}")
            print("Please install ExifTool: https://exiftool.org/")
            return False
        
        try:
            # Make a copy of the original file
            shutil.copy2(self.file_path, self.temp_path)
            
            # Build ExifTool command arguments
            exiftool_args = []
            
            # DateTime fields
            dt, error = self._get_datetime()
            if dt and not error:
                date_str = dt.strftime('%Y:%m:%d %H:%M:%S')
                exiftool_args.extend([
                    '-DateTimeOriginal=' + date_str,
                    '-CreateDate=' + date_str,
                    '-ModifyDate=' + date_str,
                    '-MediaCreateDate=' + date_str,
                    '-MediaModifyDate=' + date_str,
                    '-TrackCreateDate=' + date_str,
                    '-TrackModifyDate=' + date_str
                ])
            
            # GPS data
            gps_data, error = self._get_gps_data()
            if gps_data and not error:
                lat, lon = gps_data['latitude'], gps_data['longitude']
                exiftool_args.extend([
                    f'-GPSLatitude={abs(lat)}',
                    f'-GPSLatitudeRef={"N" if lat >= 0 else "S"}',
                    f'-GPSLongitude={abs(lon)}',
                    f'-GPSLongitudeRef={"E" if lon >= 0 else "W"}'
                ])
                
                if 'altitude' in gps_data:
                    alt = gps_data['altitude']
                    exiftool_args.extend([
                        f'-GPSAltitude={abs(alt)}',
                        f'-GPSAltitudeRef={1 if alt < 0 else 0}'
                    ])
            
            # Camera information
            camera_info, error = self._get_camera_info()
            if camera_info and not error:
                if 'make' in camera_info:
                    exiftool_args.append(f'-Make={camera_info["make"]}')
                if 'model' in camera_info:
                    exiftool_args.append(f'-Model={camera_info["model"]}')
            
            # Additional metadata
            if 'description' in self.metadata:
                exiftool_args.append(f'-ImageDescription={self.metadata["description"]}')
            if 'title' in self.metadata:
                exiftool_args.append(f'-DocumentName={self.metadata["title"]}')
            
            # Only run ExifTool if we have arguments to pass
            if exiftool_args:
                # Build the full command
                cmd = ['exiftool', '-overwrite_original']
                cmd.extend(exiftool_args)
                cmd.append(str(self.temp_path))
                
                # Run ExifTool
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    return self._handle_error(Exception(result.stderr), "running ExifTool")
                
                # Replace original file with the updated one
                self.temp_path.replace(self.file_path)
                return True
            else:
                print(f"No metadata to update for {self.file_path}")
                self._cleanup_temp_file()
                return False
                
        except Exception as e:
            return self._handle_error(e, "updating metadata with ExifTool")

class ImageMetadataHandler(MetadataHandler):
    """Handles metadata operations for image files."""
    
    def update_metadata(self) -> bool:
        """Update metadata for image files."""
        suffix = self.file_path.suffix.lower()
        
        # Handle RAW formats and AVIF that need ExifTool
        if suffix in {'.cr2', '.crw', '.nef', '.raw', '.rw2', '.dng', '.avif', '.gif'}:
            return self._update_with_exiftool()
        
        # Handle HEIC/HEIF files
        if suffix in {'.heic', '.heif'}:
            return self._update_heic_metadata()
        
        return self._update_standard_image()
    
    def _update_standard_image(self) -> bool:
        """Update metadata for standard image formats using PIL."""
        try:
            # Verify image can be opened
            try:
                with Image.open(self.file_path) as img:
                    img.load()
            except (OSError, UnidentifiedImageError) as e:
                print(f"Warning: Cannot process {self.file_path} with PIL: {e}")
                return self._update_with_exiftool()
            
            # Process the image
            with Image.open(self.file_path) as img:
                exif_dict = self._create_exif_dict()
                if not exif_dict:
                    return True  # No metadata to update
                
                # Save with new metadata
                img.save(self.temp_path, exif=piexif.dump(exif_dict))
                
                if not self.temp_path.exists():
                    return self._handle_error(Exception("Failed to create temporary file"), "saving image")
                
                self.temp_path.replace(self.file_path)
                return True
                
        except Exception as e:
            return self._handle_error(e, "updating standard image metadata")
    
    def _create_exif_dict(self) -> dict:
        """Create EXIF dictionary from metadata."""
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}
        
        # DateTime fieldss
        dt, error = self._get_datetime()
        if dt and not error:
            date_str = dt.strftime('%Y:%m:%d %H:%M:%S').encode('utf-8')
            exif_dict["0th"][piexif.ImageIFD.DateTime] = date_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_str
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = date_str
        
        # GPS data
        gps_data, error = self._get_gps_data()
        if gps_data and not error:
            self._add_gps_to_exif(exif_dict, gps_data)
        
        # Camera information
        camera_info, error = self._get_camera_info()
        if camera_info and not error:
            if 'make' in camera_info:
                exif_dict["0th"][piexif.ImageIFD.Make] = camera_info['make'].encode('utf-8')
            if 'model' in camera_info:
                exif_dict["0th"][piexif.ImageIFD.Model] = camera_info['model'].encode('utf-8')
        
        # Additional metadata
        if 'description' in self.metadata:
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = self.metadata['description'].encode('utf-8')
        if 'title' in self.metadata:
            exif_dict["0th"][piexif.ImageIFD.DocumentName] = self.metadata['title'].encode('utf-8')
        
        return exif_dict if any(exif_dict.values()) else None
    
    def _add_gps_to_exif(self, exif_dict: dict, gps_data: dict):
        """Add GPS data to EXIF dictionary."""
        lat, lon = gps_data['latitude'], gps_data['longitude']
        
        exif_dict["GPS"][piexif.GPSIFD.GPSVersionID] = (2, 2, 0, 0)
        
        # Latitude
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = 'N' if lat >= 0 else 'S'
        exif_dict["GPS"][piexif.GPSIFD.GPSLatitude] = convert_to_degrees(abs(lat))
        
        # Longitude
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = 'E' if lon >= 0 else 'W'
        exif_dict["GPS"][piexif.GPSIFD.GPSLongitude] = convert_to_degrees(abs(lon))
        
        # Altitude
        if 'altitude' in gps_data:
            alt = gps_data['altitude']
            alt_scaled = min(65535, max(0, int(alt * 10)))
            exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (alt_scaled, 10)
            exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 1 if alt < 0 else 0

    def _update_heic_metadata(self) -> bool:
        """Update metadata for HEIC files."""
        try:
            with Image.open(self.file_path) as img:
                exif_dict = self._create_exif_dict()
                if not exif_dict:
                    return True  # No metadata to update
                
                # Save as JPEG temporarily (HEIC doesn't support direct EXIF writing)
                temp_jpg = self.file_path.with_suffix('.temp.jpg')
                img.save(temp_jpg, "JPEG", exif=piexif.dump(exif_dict))
                
                # Convert back to HEIC
                with Image.open(temp_jpg) as jpg:
                    jpg.save(self.file_path, "HEIF", quality=100)
                
                # Clean up temporary file
                temp_jpg.unlink()
                return True
                
        except Exception as e:
            return self._handle_error(e, "updating HEIC metadata")

class VideoMetadataHandler(MetadataHandler):
    """Handles metadata operations for video files."""
    
    def update_metadata(self) -> bool:
        """Update metadata for video files."""
        suffix = self.file_path.suffix.lower()
        
        # Convert legacy video formats to MP4
        if suffix in LEGACY_VIDEO_FORMATS:
            return self._convert_and_update_legacy_video()
        
        if suffix in {'.lrv', '.insv'}:
            return self._update_with_exiftool()
        
        return self._update_standard_video()
    
    def _convert_and_update_legacy_video(self) -> bool:
        """Convert legacy video formats to MP4 and update metadata."""
        try:
            # Create output MP4 path
            mp4_path = self.file_path.with_suffix('.mp4')
            
            print(f"Converting {self.file_path} to MP4 format...")
            
            # Build FFmpeg command for conversion
            cmd = [
                'ffmpeg',
                '-i', str(self.file_path),
                '-c:v', 'libx264',
                '-preset', 'slow',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                str(mp4_path),
                '-y'  # Overwrite output file if it exists
            ]
            
            # Run FFmpeg conversion
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error converting {self.file_path} to MP4: {result.stderr}")
                return False
            
            # Create a new metadata handler for the MP4 file
            mp4_handler = VideoMetadataHandler(mp4_path, self.metadata)
            
            # Update metadata on the new MP4 file
            success = mp4_handler._update_standard_video()
            
            if success:
                print(f"Successfully converted and updated metadata for {self.file_path}")
                
                # Delete the original file
                try:
                    self.file_path.unlink()
                    print(f"Deleted original file: {self.file_path}")
                except Exception as e:
                    print(f"Warning: Could not delete original file {self.file_path}: {e}")
                
                return True
            else:
                print(f"Failed to update metadata for converted file {mp4_path}")
                return False
                
        except Exception as e:
            print(f"Error converting {self.file_path} to MP4: {e}")
            # Clean up MP4 file if it exists and there was an error
            if 'mp4_path' in locals() and mp4_path.exists():
                try:
                    mp4_path.unlink()
                except:
                    pass
            return False
    
    def _update_standard_video(self) -> bool:
        """Update metadata for standard video formats using ffmpeg."""
        try:
            metadata_dict = self._create_ffmpeg_metadata()
            if not metadata_dict:
                return True  # No metadata to update
            
            cmd = [
                'ffmpeg',
                '-i', str(self.file_path),
                '-c', 'copy',
                '-map_metadata', '0',
                '-movflags', 'use_metadata_tags'
            ]

            # Add metadata arguments
            for key, value in metadata_dict.items():
                cmd.extend(['-metadata', f'{key}={value}'])
            
            cmd.append(str(self.temp_path))
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"FFmpeg error, trying ExifTool as fallback: {result.stderr}")
                return self._update_with_exiftool()
            
            self.temp_path.replace(self.file_path)
            return True
            
        except Exception as e:
            return self._handle_error(e, "updating standard video metadata")
    
    def _create_ffmpeg_metadata(self) -> dict:
        """Create metadata dictionary for ffmpeg."""
        metadata_dict = {}
        
        # DateTime fields
        dt, error = self._get_datetime()
        if dt and not error:
            if self.file_path.suffix.lower() in {'.avi', '.mpg', '.mpeg', '.mpe'}:
                metadata_dict["date"] = dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
                metadata_dict["recording_time"] = dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
            else:
                metadata_dict["creation_time"] = dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')
        
        # Camera information
        camera_info, error = self._get_camera_info()
        if camera_info and not error:
            if 'make' in camera_info:
                metadata_dict["make"] = camera_info['make']
            if 'model' in camera_info:
                metadata_dict["model"] = camera_info['model']
        
        # GPS data
        gps_data, error = self._get_gps_data()
        if gps_data and not error:
            location = f"{gps_data['latitude']},{gps_data['longitude']}"
            if 'altitude' in gps_data:
                location += f",{gps_data['altitude']}"
            metadata_dict["location"] = location
        
        return metadata_dict if metadata_dict else None

def get_media_files(folder: Path) -> Dict[str, dict]:
    media_files = {}
    for ext in SUPPORTED_EXTENSIONS:
        # Remove the dot and get all case variations
        ext_without_dot = ext[1:]  # remove the leading dot
        # Create case-insensitive glob patterns by using * for each character
        pattern = "*." + "".join(f"[{c.lower()}{c.upper()}]" for c in ext_without_dot)
        
        # Search using the case-insensitive pattern
        for file_path in folder.glob(pattern):
            if file_path.is_file() and file_path.stat().st_size > 0:
                stem = file_path.stem.lower()
                
                file_info = {
                    'original_name': file_path.name,
                    'base_stem': stem,
                    'number': 0,
                    'extension': file_path.suffix.lower(),
                    'path': file_path,
                    'media_type': 'IMAGE' if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS else 'VIDEO'
                }

                number_match = re.search(r'(.*?)\((\d+)\)$', stem)
                if number_match:
                    file_info['number'] = int(number_match.group(2))
                    file_info['base_stem'] = number_match.group(1).strip()
                
                media_files[file_path.name.lower()] = file_info
    return media_files

def get_json_files(folder: Path) -> Dict[str, dict]:
    json_files = {}
    for json_path in folder.glob("*.json"):
        if not json_path.is_file() or json_path.stat().st_size == 0:
            continue
        
        # Get the name without the .json extension using Path's stem property
        # For example: "image.jpg.supplemental-metadata.json" -> "image.jpg.supplemental-metadata"
        name_without_json = json_path.stem
        
        # Split the remaining name into parts by dots
        name_parts = name_without_json.split('.')
        
        file_info = {
            'original_name': json_path.name,
            'path': json_path,
            'name_parts': name_parts,
            'matched_media_file': None,
            'number': 0,
        }
        
        # Check each part for a potential extension
        for i, part in enumerate(name_parts):
            detected_extension = '.' + part.lower()
            if detected_extension in SUPPORTED_EXTENSIONS:
                file_info['detected_extension'] = detected_extension
                # Everything after the extension (if any) becomes the suffix
                if i < len(name_parts) - 1:
                    file_info['metadata_label'] = name_parts[i+1]
                break
        
        # The firt part is the base name
        file_info['base_name'] = name_parts[0]

        # Verify if there is a number in the base name
        number_match = re.search(r'\((\d+)\)', file_info['base_name'])

        # If there is a number, extract it and remove it from the base_name
        if number_match:
            file_info['number'] = int(number_match.group(1))
            # Remove the (number) from base_name
            file_info['base_name'] = re.sub(r'\(\d+\)', '', file_info['base_name']).strip()
        else:
            # If there is no number in the base name, there is potential number in the metadata label (aka supplemental-metadata(1))
            if 'metadata_label' in file_info:
                number_match = re.search(r'\((\d+)\)', file_info['metadata_label'])
                if number_match:
                    file_info['number'] = int(number_match.group(1))
                    # Remove the (number) from metadata_label
                    file_info['metadata_label'] = re.sub(r'\(\d+\)', '', file_info['metadata_label']).strip()
        
        json_files[json_path.name.lower()] = file_info
    return json_files

def process_json_file(json_info: dict, dry_run: bool = False) -> None:
    """Process a single JSON file and its matched media file."""
    try:
        matched_media = json_info['matched_media_file']
        if not matched_media:
            print(f"{json_info['original_name']} -> No match found")
            return

        if dry_run:
            print(f"Would update metadata for {matched_media['original_name']}")
            return

        print(f"{matched_media['original_name']} -> Updating metadata")
        
        media_file = json_info['matched_media_file']
        json_metadata = get_metadata_from_json(json_info['path'])
        
        handler_class = ImageMetadataHandler if media_file['media_type'] == 'IMAGE' else VideoMetadataHandler
        handler = handler_class(media_file['path'], json_metadata)
        
        success = handler.update_metadata()
        
        # Delete the JSON file if metadata was successfully updated
        if success:
            try:
                json_info['path'].unlink()
            except Exception as e:
                print(f"Error deleting JSON file {json_info['path']}: {e}")
    except Exception as e:
        print(f"Error processing {json_info['original_name']}: {e}")

def process_folder(folder: Path, dry_run: bool = True) -> None:
    print(f"\nProcessing folder: {folder}")
    
    # Get all files in this folder
    media_files = get_media_files(folder)
    json_files = get_json_files(folder)
    
    print(f"Found {len(media_files)} media files and {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_name, json_info in json_files.items():
        matches = [media_info for media_info in media_files.values() 
                    if (json_info['base_name'].lower() == media_info['base_stem'].lower() and
                        ('detected_extension' not in json_info or 
                         json_info['detected_extension'].lower() == media_info['extension'].lower()))]
        if matches:
            if len(matches) == 1:
                # Single match - use it directly
                match = matches[0]
                json_info['matched_media_file'] = match
            else:
                # Multiple matches - try to find one with matching number
                number_matches = [m for m in matches if m['number'] == json_info['number']]
                if number_matches:
                    match = number_matches[0]
                    json_info['matched_media_file'] = match

     # Process again, but this time use the IN istead of ==
    for json_name, json_info in json_files.items():
        if json_info['matched_media_file']:
            continue

        matches = [media_info for media_info in media_files.values() 
                    if (json_info['base_name'].lower() in media_info['base_stem'].lower() and
                        ('detected_extension' not in json_info or 
                         json_info['detected_extension'].lower() == media_info['extension'].lower()))]
        if matches:
            if len(matches) == 1:
                # Single match - use it directly
                match = matches[0]
                json_info['matched_media_file'] = match
            else:
                # Multiple matches - try to find one with matching number
                number_matches = [m for m in matches if m['number'] == json_info['number']]
                if number_matches:
                    match = number_matches[0]
                    json_info['matched_media_file'] = match

    if not dry_run:
        # Create a process pool with number of cores minus 1 (leave one for system)
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Processing files using {num_processes} processes")
        
        # Convert dictionary values to list for parallel processing
        json_files_list = list(json_files.values())
        
        # Create a partial function with dry_run parameter
        process_func = partial(process_json_file, dry_run=dry_run)
        
        # Process files in parallel
        with multiprocessing.Pool(num_processes) as pool:
            pool.map(process_func, json_files_list)

def update_metadata(json_info: dict) -> None:
    """Update metadata for a media file - kept for backwards compatibility."""
    process_json_file(json_info, dry_run=False)

def parse_timestamp(timestamp_value: str) -> datetime:
    """Parse timestamp from either Unix timestamp or ISO format."""
    try:
        # Try parsing as Unix timestamp first
        return datetime.fromtimestamp(int(timestamp_value))
    except ValueError:
        # If that fails, try ISO format
        return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))

def get_metadata_from_json(json_path: Path) -> dict:
    """Read and parse metadata from a JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            return metadata
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {json_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return {}

def convert_to_degrees(value: float) -> tuple:
    """Convert decimal GPS coordinates into degrees format for EXIF."""
    try:
        # Calculate degrees, minutes, seconds
        d = int(abs(value))
        m = int((abs(value) - d) * 60)
        s = int((abs(value) - d - m/60) * 3600)
        
        # Return as rational numbers with denominator of 1
        return ((d, 1), (m, 1), (s, 1))
    except Exception as e:
        print(f"Error converting GPS coordinate {value}: {e}")
        return ((0, 1), (0, 1), (0, 1))

def process_directory(directory: Path, dry_run: bool = True) -> None:
    """
    Recursively process a directory and all its subdirectories.
    
    Args:
        directory: Path to the directory to process
        dry_run: If True, don't actually update any files
    """
    # Process current directory
    print(f"\nProcessing directory: {directory}")
    process_folder(directory, dry_run)
    
    # Recursively process each subdirectory
    for folder_path in directory.iterdir():
        if folder_path.is_dir():
            process_directory(folder_path, dry_run)  # Recursive call

def main():
    parser = argparse.ArgumentParser(description='Find potential matches between JSON and media files')
    parser.add_argument('directory', help='Directory containing media and JSON files')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Only analyze files without modifying them')
    
    args = parser.parse_args()
    directory = Path(args.directory).resolve()
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    process_directory(directory, args.dry_run)

if __name__ == '__main__':
    main()