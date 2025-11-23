"""
SPDX-FileCopyrightText: 2025 TEMPL Team
SPDX-License-Identifier: MIT

RDKit Compatibility Layer

Provides compatibility wrappers for RDKit API changes across versions.
Prevents common errors like the Morgan fingerprint generator API mismatch.
"""

from typing import Optional
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Some functionality will be limited.")


def get_morgan_generator(
    radius: int = 2,
    fp_size: int = 2048,
    use_features: bool = False,
    use_chirality: bool = False,
    use_bond_types: bool = True
) -> Optional['rdFingerprintGenerator.FingerprintGenerator64']:
    """
    Get Morgan fingerprint generator with correct API for current RDKit version.
    
    This function handles the API change where 'nBits' was renamed to 'fpSize'
    in newer RDKit versions, preventing the common ArgumentError.
    
    Args:
        radius: Morgan fingerprint radius (default: 2)
        fp_size: Fingerprint size in bits (default: 2048)
        use_features: Use feature-based Morgan fingerprints
        use_chirality: Include chirality information
        use_bond_types: Use bond type information
        
    Returns:
        Morgan fingerprint generator object
        
    Raises:
        ImportError: If RDKit is not available
        
    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> gen = get_morgan_generator(radius=2, fp_size=2048)
        >>> fp = gen.GetFingerprint(mol)
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required but not installed")
    
    try:
        # Try new API first (fpSize parameter)
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=fp_size,
            includeChirality=use_chirality,
            useBondTypes=use_bond_types
        )
        return generator
    except TypeError:
        # Fallback to old API (nBits parameter) for older RDKit versions
        try:
            # Note: This is for backwards compatibility only
            # Modern RDKit (2024+) uses fpSize
            generator = rdFingerprintGenerator.GetMorganGenerator(
                radius=radius,
                nBits=fp_size,  # Old parameter name
                includeChirality=use_chirality,
                useBondTypes=use_bond_types
            )
            warnings.warn(
                "Using legacy RDKit API. Consider upgrading to RDKit 2024+",
                DeprecationWarning
            )
            return generator
        except Exception as e:
            raise RuntimeError(
                f"Failed to create Morgan fingerprint generator. "
                f"RDKit version may be incompatible. Error: {e}"
            )


def get_rdkit_fingerprint(
    mol: 'Chem.Mol',
    radius: int = 2,
    fp_size: int = 2048,
    use_chirality: bool = False
) -> Optional['rdFingerprintGenerator.UIntSparseIntVect']:
    """
    Generate Morgan fingerprint for a molecule with automatic API compatibility.
    
    Args:
        mol: RDKit molecule object
        radius: Morgan fingerprint radius
        fp_size: Fingerprint size in bits
        use_chirality: Include chirality information
        
    Returns:
        Morgan fingerprint as bit vector
        
    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")
        >>> fp = get_rdkit_fingerprint(mol, radius=2, fp_size=2048)
    """
    if mol is None:
        return None
    
    generator = get_morgan_generator(
        radius=radius,
        fp_size=fp_size,
        use_chirality=use_chirality
    )
    
    return generator.GetFingerprint(mol)


def check_rdkit_version() -> tuple:
    """
    Check RDKit version and return as tuple.
    
    Returns:
        Tuple of (major, minor, patch) version numbers
        
    Example:
        >>> major, minor, patch = check_rdkit_version()
        >>> print(f"RDKit {major}.{minor}.{patch}")
    """
    if not RDKIT_AVAILABLE:
        return (0, 0, 0)
    
    import rdkit
    version_str = rdkit.__version__
    
    # Parse version string (e.g., "2024.09.6")
    parts = version_str.split('.')
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        return (major, minor, patch)
    except (ValueError, IndexError):
        warnings.warn(f"Could not parse RDKit version: {version_str}")
        return (0, 0, 0)


def is_rdkit_modern() -> bool:
    """
    Check if RDKit version is modern (2024+).
    
    Returns:
        True if RDKit is version 2024 or newer
    """
    major, _, _ = check_rdkit_version()
    return major >= 2024


# Module-level version check on import
if RDKIT_AVAILABLE:
    _version = check_rdkit_version()
    if _version[0] > 0:
        print(f"RDKit compatibility layer loaded (RDKit {_version[0]}.{_version[1]}.{_version[2]})")
        if not is_rdkit_modern():
            warnings.warn(
                f"RDKit version {_version[0]}.{_version[1]}.{_version[2]} is outdated. "
                "Consider upgrading to RDKit 2024+ for best compatibility.",
                UserWarning
            )
