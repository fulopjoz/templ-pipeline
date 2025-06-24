"""
TEMPL Pipeline Molecular Descriptors for FAIR

This module provides comprehensive molecular descriptor calculation
and metadata generation for FAIR-compliant biological workflows.

Key Features:
- Comprehensive molecular property calculation
- Chemical space characterization
- Drug-like property assessment
- ADMET-relevant descriptors
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MolecularProperties:
    """Container for molecular properties and descriptors."""
    molecular_formula: str
    molecular_weight: float
    exact_mass: float
    logp: float
    tpsa: float
    hbd: int  # Hydrogen bond donors
    hba: int  # Hydrogen bond acceptors
    rotatable_bonds: int
    ring_count: int
    aromatic_rings: int
    formal_charge: int
    
@dataclass
class DrugLikeProperties:
    """Drug-likeness assessment properties."""
    lipinski_violations: int
    qed_score: float
    sa_score: float  # Synthetic accessibility
    ro5_compliant: bool
    veber_compliant: bool
    ghose_compliant: bool
    
@dataclass
class ChemicalFingerprints:
    """Chemical fingerprints for similarity assessment."""
    morgan_fp: str  # Base64 encoded
    rdkit_fp: str
    maccs_keys: str
    tanimoto_self: float

class MolecularDescriptorEngine:
    """
    Engine for calculating comprehensive molecular descriptors
    and chemical properties for FAIR metadata.
    """
    
    def __init__(self):
        """Initialize the molecular descriptor engine."""
        self._init_rdkit()
        
    def _init_rdkit(self):
        """Initialize RDKit components."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen
            from rdkit.Chem import Lipinski, QED
            from rdkit.Chem.Fingerprints import FingerprintMols
            from rdkit.Chem import MACCSkeys
            
            self.Chem = Chem
            self.Descriptors = Descriptors
            self.rdMolDescriptors = rdMolDescriptors
            self.Crippen = Crippen
            self.Lipinski = Lipinski
            self.QED = QED
            self.FingerprintMols = FingerprintMols
            self.MACCSkeys = MACCSkeys
            
        except ImportError as e:
            logger.error(f"RDKit not available: {e}")
            raise ImportError("RDKit is required for molecular descriptor calculation")
    
    def calculate_molecular_properties(self, smiles: str) -> Optional[MolecularProperties]:
        """
        Calculate basic molecular properties from SMILES.
        
        Args:
            smiles: SMILES string
            
        Returns:
            MolecularProperties object or None if calculation fails
        """
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None
            
            return MolecularProperties(
                molecular_formula=self.rdMolDescriptors.CalcMolFormula(mol),
                molecular_weight=self.Descriptors.MolWt(mol),
                exact_mass=self.Descriptors.ExactMolWt(mol),
                logp=self.Crippen.MolLogP(mol),
                tpsa=self.Descriptors.TPSA(mol),
                hbd=self.Descriptors.NumHDonors(mol),
                hba=self.Descriptors.NumHAcceptors(mol),
                rotatable_bonds=self.Descriptors.NumRotatableBonds(mol),
                ring_count=self.rdMolDescriptors.CalcNumRings(mol),
                aromatic_rings=self.rdMolDescriptors.CalcNumAromaticRings(mol),
                formal_charge=self.Chem.rdmolops.GetFormalCharge(mol)
            )
            
        except Exception as e:
            logger.error(f"Error calculating molecular properties for {smiles}: {e}")
            return None
    
    def assess_drug_likeness(self, smiles: str) -> Optional[DrugLikeProperties]:
        """
        Assess drug-likeness properties.
        
        Args:
            smiles: SMILES string
            
        Returns:
            DrugLikeProperties object or None if calculation fails
        """
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Lipinski Rule of 5 violations
            lipinski_violations = self._count_lipinski_violations(mol)
            
            # QED score (drug-likeness)
            qed_score = self.QED.qed(mol)
            
            # Synthetic accessibility (simplified)
            sa_score = self._calculate_sa_score(mol)
            
            # Rule compliance
            ro5_compliant = lipinski_violations == 0
            veber_compliant = self._check_veber_compliance(mol)
            ghose_compliant = self._check_ghose_compliance(mol)
            
            return DrugLikeProperties(
                lipinski_violations=lipinski_violations,
                qed_score=qed_score,
                sa_score=sa_score,
                ro5_compliant=ro5_compliant,
                veber_compliant=veber_compliant,
                ghose_compliant=ghose_compliant
            )
            
        except Exception as e:
            logger.error(f"Error assessing drug-likeness for {smiles}: {e}")
            return None
    
    def generate_fingerprints(self, smiles: str) -> Optional[ChemicalFingerprints]:
        """
        Generate chemical fingerprints for similarity assessment.
        
        Args:
            smiles: SMILES string
            
        Returns:
            ChemicalFingerprints object or None if calculation fails
        """
        try:
            mol = self.Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Morgan fingerprint (ECFP4)
            morgan_fp = self.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            morgan_str = self._bitvect_to_base64(morgan_fp)
            
            # RDKit fingerprint
            rdkit_fp = self.FingerprintMols.FingerprintMol(mol)
            rdkit_str = self._bitvect_to_base64(rdkit_fp)
            
            # MACCS keys
            maccs_fp = self.MACCSkeys.GenMACCSKeys(mol)
            maccs_str = self._bitvect_to_base64(maccs_fp)
            
            return ChemicalFingerprints(
                morgan_fp=morgan_str,
                rdkit_fp=rdkit_str,
                maccs_keys=maccs_str,
                tanimoto_self=1.0  # Self-similarity is always 1.0
            )
            
        except Exception as e:
            logger.error(f"Error generating fingerprints for {smiles}: {e}")
            return None
    
    def _count_lipinski_violations(self, mol) -> int:
        """Count Lipinski Rule of 5 violations."""
        violations = 0
        
        if self.Descriptors.MolWt(mol) > 500:
            violations += 1
        if self.Crippen.MolLogP(mol) > 5:
            violations += 1
        if self.Descriptors.NumHDonors(mol) > 5:
            violations += 1
        if self.Descriptors.NumHAcceptors(mol) > 10:
            violations += 1
            
        return violations
    
    def _calculate_sa_score(self, mol) -> float:
        """Calculate simplified synthetic accessibility score."""
        # Simplified SA score based on complexity indicators
        try:
            # Basic complexity indicators
            ring_complexity = self.rdMolDescriptors.CalcNumRings(mol) * 0.1
            rotatable_complexity = self.Descriptors.NumRotatableBonds(mol) * 0.05
            heteroatom_complexity = self.rdMolDescriptors.CalcNumHeteroatoms(mol) * 0.1
            
            # Simple SA score (lower is better, range 0-10)
            sa_score = min(10.0, ring_complexity + rotatable_complexity + heteroatom_complexity)
            return round(sa_score, 2)
            
        except:
            return 5.0  # Default medium complexity
    
    def _check_veber_compliance(self, mol) -> bool:
        """Check Veber rule compliance."""
        tpsa = self.Descriptors.TPSA(mol)
        rotatable = self.Descriptors.NumRotatableBonds(mol)
        
        return tpsa <= 140 and rotatable <= 10
    
    def _check_ghose_compliance(self, mol) -> bool:
        """Check Ghose filter compliance."""
        mw = self.Descriptors.MolWt(mol)
        logp = self.Crippen.MolLogP(mol)
        
        return 160 <= mw <= 480 and -0.4 <= logp <= 5.6
    
    def _bitvect_to_base64(self, bitvect) -> str:
        """Convert RDKit bitvector to base64 string."""
        try:
            import base64
            # Convert bitvector to bytes then to base64
            bit_string = bitvect.ToBitString()
            bit_bytes = int(bit_string, 2).to_bytes((len(bit_string) + 7) // 8, 'big')
            return base64.b64encode(bit_bytes).decode('ascii')
        except:
            return ""

def calculate_comprehensive_descriptors(smiles: str) -> Dict[str, Any]:
    """
    Calculate comprehensive molecular descriptors for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary containing all calculated descriptors
    """
    engine = MolecularDescriptorEngine()
    
    result = {
        "smiles": smiles,
        "calculation_success": False,
        "molecular_properties": None,
        "drug_likeness": None,
        "fingerprints": None
    }
    
    try:
        # Calculate molecular properties
        mol_props = engine.calculate_molecular_properties(smiles)
        if mol_props:
            result["molecular_properties"] = mol_props.__dict__
        
        # Assess drug-likeness
        drug_props = engine.assess_drug_likeness(smiles)
        if drug_props:
            result["drug_likeness"] = drug_props.__dict__
        
        # Generate fingerprints
        fingerprints = engine.generate_fingerprints(smiles)
        if fingerprints:
            result["fingerprints"] = fingerprints.__dict__
        
        result["calculation_success"] = any([mol_props, drug_props, fingerprints])
        
    except Exception as e:
        logger.error(f"Error in comprehensive descriptor calculation: {e}")
    
    return result

def calculate_template_similarity(query_smiles: str, template_smiles: str) -> Dict[str, float]:
    """
    Calculate molecular similarity between query and template molecules.
    
    Args:
        query_smiles: Query molecule SMILES
        template_smiles: Template molecule SMILES
        
    Returns:
        Dictionary with similarity metrics
    """
    engine = MolecularDescriptorEngine()
    
    try:
        from rdkit import DataStructs
        
        query_mol = engine.Chem.MolFromSmiles(query_smiles)
        template_mol = engine.Chem.MolFromSmiles(template_smiles)
        
        if not query_mol or not template_mol:
            return {"error": "Invalid SMILES"}
        
        # Generate fingerprints
        query_morgan = engine.rdMolDescriptors.GetMorganFingerprintAsBitVect(query_mol, 2)
        template_morgan = engine.rdMolDescriptors.GetMorganFingerprintAsBitVect(template_mol, 2)
        
        query_rdkit = engine.FingerprintMols.FingerprintMol(query_mol)
        template_rdkit = engine.FingerprintMols.FingerprintMol(template_mol)
        
        # Calculate similarities
        morgan_tanimoto = DataStructs.TanimotoSimilarity(query_morgan, template_morgan)
        rdkit_tanimoto = DataStructs.TanimotoSimilarity(query_rdkit, template_rdkit)
        
        return {
            "morgan_tanimoto": round(morgan_tanimoto, 4),
            "rdkit_tanimoto": round(rdkit_tanimoto, 4),
            "average_similarity": round((morgan_tanimoto + rdkit_tanimoto) / 2, 4)
        }
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return {"error": str(e)} 