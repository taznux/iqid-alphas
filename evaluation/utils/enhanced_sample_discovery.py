#!/usr/bin/env python3
"""
Enhanced Sample Discovery for iQID Dataset

Groups samples by common base ID (e.g., D1M1) and tracks tissue separation
from single raw images to multiple tissue types.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TissueSample:
    """Represents a single tissue sample"""
    base_id: str  # e.g., "D1M1"
    tissue_type: str  # e.g., "tumor", "kidney_L", "kidney_R"
    study_type: str  # "3D" or "Sequential"
    part_number: Optional[str]  # e.g., "P1" for acquisition part
    raw_files: List[Path]
    segmented_files: List[Path]
    aligned_files: List[Path]
    he_files: List[Path]  # H&E reference files
    sample_path: Path
    
    @property
    def full_id(self) -> str:
        """Full sample identifier including tissue type"""
        if self.part_number:
            return f"{self.base_id}({self.part_number})_{self.tissue_type}"
        return f"{self.base_id}_{self.tissue_type}"


@dataclass
class SampleGroup:
    """Group of tissue samples that should share the same raw image"""
    base_id: str
    study_type: str
    part_number: Optional[str]
    tissues: Dict[str, TissueSample]  # tissue_type -> TissueSample
    shared_raw_files: List[Path]  # Raw files that should be shared across tissues
    
    @property
    def group_id(self) -> str:
        """Unique identifier for this sample group"""
        if self.part_number:
            return f"{self.base_id}({self.part_number})_{self.study_type}"
        return f"{self.base_id}_{self.study_type}"
    
    def get_expected_tissues(self) -> List[str]:
        """Get expected tissue types based on study type and naming patterns"""
        if "tumor" in str(self.tissues.keys()).lower():
            return ["tumor", "kidney_L", "kidney_R"]
        else:
            return ["kidney_L", "kidney_R"]
    
    def is_complete(self) -> bool:
        """Check if all expected tissues are present"""
        expected = set(self.get_expected_tissues())
        actual = set(self.tissues.keys())
        return expected.issubset(actual)
    
    def get_missing_tissues(self) -> List[str]:
        """Get list of missing tissue types"""
        expected = set(self.get_expected_tissues())
        actual = set(self.tissues.keys())
        return list(expected - actual)


class EnhancedSampleDiscovery:
    """Enhanced sample discovery that groups by common ID and tracks tissue separation"""
    
    def __init__(self):
        self.naming_patterns = {
            'base_id': r'D(\d+)M(\d+)',  # Captures day and mouse number
            'part': r'\(P(\d+)\)',       # Captures part number (acquisition)
            'tissue': r'_([LR]|tumor)$', # Captures tissue type (L, R, or tumor)
        }
    
    def parse_sample_name(self, sample_name: str) -> Dict[str, Optional[str]]:
        """
        Parse iQID sample name into components
        
        Examples:
        - "D1M1_L" -> {"base_id": "D1M1", "tissue": "kidney_L", "part": None}
        - "D1M1(P1)_tumor" -> {"base_id": "D1M1", "tissue": "tumor", "part": "P1"}
        """
        # Extract base ID (day + mouse)
        base_match = re.search(self.naming_patterns['base_id'], sample_name)
        if not base_match:
            return {"base_id": None, "tissue": None, "part": None}
        
        base_id = base_match.group(0)  # e.g., "D1M1"
        
        # Extract part number if present
        part_match = re.search(self.naming_patterns['part'], sample_name)
        part = part_match.group(1) if part_match else None
        
        # Extract tissue type
        tissue_match = re.search(self.naming_patterns['tissue'], sample_name)
        if tissue_match:
            tissue_code = tissue_match.group(1)
            if tissue_code == 'L':
                tissue_type = 'kidney_L'
            elif tissue_code == 'R':
                tissue_type = 'kidney_R'
            elif tissue_code == 'tumor':
                tissue_type = 'tumor'
            else:
                tissue_type = tissue_code
        else:
            tissue_type = None
        
        return {
            "base_id": base_id,
            "tissue": tissue_type,
            "part": f"P{part}" if part else None
        }
    
    def discover_sample_groups(self, data_root: Path) -> Dict[str, SampleGroup]:
        """
        Discover and group samples by common base ID
        
        Returns:
            Dictionary mapping group_id -> SampleGroup
        """
        groups = defaultdict(lambda: defaultdict(list))  # base_id -> study_type -> samples
        
        # Scan both 3D and Sequential directories
        for study_type in ["3D", "Sequential"]:
            study_path = data_root / study_type
            if not study_path.exists():
                continue
            
            # Scan tissue type directories
            for tissue_dir in ["tumor", "kidney"]:
                tissue_path = study_path / tissue_dir
                if not tissue_path.exists():
                    continue
                
                # Find sample directories
                for sample_dir in tissue_path.iterdir():
                    if not sample_dir.is_dir():
                        continue
                    
                    # Parse sample name
                    parsed = self.parse_sample_name(sample_dir.name)
                    if not parsed["base_id"] or not parsed["tissue"]:
                        continue
                    
                    # Create TissueSample
                    tissue_sample = self._create_tissue_sample(
                        sample_dir, parsed, study_type
                    )
                    
                    # Group by base_id and study_type
                    group_key = f"{parsed['base_id']}_{study_type}"
                    if parsed["part"]:
                        group_key += f"_{parsed['part']}"
                    
                    groups[group_key].append(tissue_sample)
        
        # Convert to SampleGroup objects
        sample_groups = {}
        for group_key, tissue_samples in groups.items():
            if not tissue_samples:
                continue
            
            # Extract group info from first sample
            first_sample = tissue_samples[0]
            
            # Create SampleGroup
            group = SampleGroup(
                base_id=first_sample.base_id,
                study_type=first_sample.study_type,
                part_number=first_sample.part_number,
                tissues={},
                shared_raw_files=[]
            )
            
            # Add tissues to group
            for tissue_sample in tissue_samples:
                group.tissues[tissue_sample.tissue_type] = tissue_sample
            
            # Find shared raw files (should be the same across all tissues in group)
            group.shared_raw_files = self._find_shared_raw_files(tissue_samples)
            
            sample_groups[group.group_id] = group
        
        return sample_groups
    
    def _create_tissue_sample(self, sample_dir: Path, parsed: Dict, study_type: str) -> TissueSample:
        """Create a TissueSample from directory and parsed name"""
        
        # Find files in different processing stages
        raw_files = list(sample_dir.glob("raw/*.tif")) if (sample_dir / "raw").exists() else []
        segmented_files = list(sample_dir.glob("segmented/*.tif")) if (sample_dir / "segmented").exists() else []
        aligned_files = list(sample_dir.glob("aligned/*.tif")) if (sample_dir / "aligned").exists() else []
        
        # For H&E files, look in the corresponding H&E directory structure
        he_files = self._find_he_files(sample_dir, parsed, study_type)
        
        return TissueSample(
            base_id=parsed["base_id"],
            tissue_type=parsed["tissue"],
            study_type=study_type,
            part_number=parsed["part"],
            raw_files=raw_files,
            segmented_files=segmented_files,
            aligned_files=aligned_files,
            he_files=he_files,
            sample_path=sample_dir
        )
    
    def _find_he_files(self, sample_dir: Path, parsed: Dict, study_type: str) -> List[Path]:
        """Find corresponding H&E files for a sample"""
        # Look in the H&E directory structure
        data_root = sample_dir.parent.parent.parent.parent  # Go up to DataPush1/ReUpload level
        
        he_path = data_root / "HE" / study_type
        if parsed["tissue"] in ["kidney_L", "kidney_R"]:
            he_path = he_path / "kidney" / f"{parsed['base_id']}_{parsed['tissue'][-1]}"
        else:
            he_path = he_path / "tumor" / f"{parsed['base_id']}_tumor"
        
        if he_path.exists():
            return list(he_path.glob("*.tif"))
        return []
    
    def _find_shared_raw_files(self, tissue_samples: List[TissueSample]) -> List[Path]:
        """Find raw files that should be shared across tissue samples"""
        # In theory, all tissues from the same base_id should share raw files
        # But in practice, we need to check which files actually exist
        
        all_raw_files = set()
        for tissue_sample in tissue_samples:
            all_raw_files.update(tissue_sample.raw_files)
        
        return list(all_raw_files)
    
    def validate_sample_groups(self, sample_groups: Dict[str, SampleGroup]) -> Dict[str, List[str]]:
        """
        Validate sample groups and return issues found
        
        Returns:
            Dictionary mapping group_id -> list of validation issues
        """
        validation_issues = {}
        
        for group_id, group in sample_groups.items():
            issues = []
            
            # Check if group is complete
            if not group.is_complete():
                missing = group.get_missing_tissues()
                issues.append(f"Missing tissues: {missing}")
            
            # Check if tissues have consistent file counts
            file_counts = {}
            for tissue_type, tissue_sample in group.tissues.items():
                file_counts[tissue_type] = {
                    'raw': len(tissue_sample.raw_files),
                    'segmented': len(tissue_sample.segmented_files),
                    'aligned': len(tissue_sample.aligned_files)
                }
            
            # Check for inconsistent raw file counts (should be same if shared)
            raw_counts = [counts['raw'] for counts in file_counts.values()]
            if len(set(raw_counts)) > 1:
                issues.append(f"Inconsistent raw file counts: {file_counts}")
            
            # Check if any tissue has no files
            for tissue_type, counts in file_counts.items():
                if all(count == 0 for count in counts.values()):
                    issues.append(f"Tissue {tissue_type} has no files")
            
            if issues:
                validation_issues[group_id] = issues
        
        return validation_issues
    
    def print_discovery_summary(self, sample_groups: Dict[str, SampleGroup]):
        """Print a summary of discovered sample groups"""
        print(f"\n=== Sample Group Discovery Summary ===")
        print(f"Total groups found: {len(sample_groups)}")
        
        # Group by study type
        by_study = defaultdict(list)
        for group in sample_groups.values():
            by_study[group.study_type].append(group)
        
        for study_type, groups in by_study.items():
            print(f"\n{study_type} Study:")
            print(f"  Groups: {len(groups)}")
            
            for group in sorted(groups, key=lambda g: g.base_id):
                tissues = list(group.tissues.keys())
                completeness = "✓" if group.is_complete() else "✗"
                shared_raw = len(group.shared_raw_files)
                
                print(f"    {group.group_id} {completeness}")
                print(f"      Tissues: {tissues}")
                print(f"      Shared raw files: {shared_raw}")
                
                if not group.is_complete():
                    missing = group.get_missing_tissues()
                    print(f"      Missing: {missing}")


def test_enhanced_discovery():
    """Test the enhanced sample discovery"""
    discovery = EnhancedSampleDiscovery()
    
    # Test parsing
    test_names = [
        "D1M1_L",
        "D1M1_R", 
        "D1M1(P1)_tumor",
        "D7M2_L",
        "D7M2(P2)_tumor"
    ]
    
    print("=== Name Parsing Test ===")
    for name in test_names:
        parsed = discovery.parse_sample_name(name)
        print(f"{name} -> {parsed}")
    
    # Test on actual data if available
    data_root = Path("/home/wxc151/iqid-alphas/data")
    if data_root.exists():
        for dataset in ["DataPush1", "ReUpload"]:
            dataset_path = data_root / dataset / "iQID_reupload" / "iQID" if dataset == "ReUpload" else data_root / dataset
            if dataset_path.exists():
                print(f"\n=== Testing on {dataset} ===")
                groups = discovery.discover_sample_groups(dataset_path)
                discovery.print_discovery_summary(groups)
                
                # Validate
                issues = discovery.validate_sample_groups(groups)
                if issues:
                    print(f"\n=== Validation Issues ===")
                    for group_id, group_issues in issues.items():
                        print(f"{group_id}:")
                        for issue in group_issues:
                            print(f"  - {issue}")


if __name__ == "__main__":
    test_enhanced_discovery()
