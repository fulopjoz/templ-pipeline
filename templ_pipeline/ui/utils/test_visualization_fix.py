#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test script to verify molecule visualization fixes

This script tests the improved molecule storage and retrieval system
to ensure visualization issues are resolved.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def test_molecule_storage_and_retrieval():
    """Test molecule storage and retrieval with the new system"""
    try:
        from rdkit import Chem
        from templ_pipeline.ui.core.session_manager import SessionManager
        from templ_pipeline.ui.config.settings import get_config
        from templ_pipeline.ui.config.constants import SESSION_KEYS
        from templ_pipeline.ui.utils.visualization_utils import (
            get_molecule_from_session,
            create_mcs_molecule_from_info
        )
        
        print("Testing molecule visualization fixes...")
        
        # Create test setup
        config = get_config()
        session = SessionManager(config)
        session.initialize()
        
        # Test 1: SMILES molecule storage and retrieval
        print("\n1. Testing SMILES molecule storage...")
        test_smiles = "CCO"  # Ethanol
        test_mol = Chem.MolFromSmiles(test_smiles)
        test_mol.SetProp("original_smiles", test_smiles)
        
        # Store molecule
        session.set(SESSION_KEYS["QUERY_MOL"], test_mol)
        session.set(SESSION_KEYS["INPUT_SMILES"], test_smiles)
        
        # Retrieve molecule using utility function
        retrieved_mol = get_molecule_from_session(
            session, SESSION_KEYS["QUERY_MOL"], fallback_smiles=test_smiles
        )
        
        if retrieved_mol:
            print(f"✅ Successfully retrieved query molecule: {Chem.MolToSmiles(retrieved_mol)}")
        else:
            print("❌ Failed to retrieve query molecule")
            return False
        
        # Test 2: Template molecule with metadata
        print("\n2. Testing template molecule storage...")
        template_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        template_mol = Chem.MolFromSmiles(template_smiles)
        template_mol.SetProp("template_pdb_id", "2hyy")
        template_mol.SetProp("template_smiles", template_smiles)
        
        session.set(SESSION_KEYS["TEMPLATE_USED"], template_mol)
        
        # Retrieve template molecule
        retrieved_template = get_molecule_from_session(
            session, SESSION_KEYS["TEMPLATE_USED"]
        )
        
        if retrieved_template:
            print(f"✅ Successfully retrieved template molecule: {Chem.MolToSmiles(retrieved_template)}")
        else:
            print("❌ Failed to retrieve template molecule")
            return False
        
        # Test 3: MCS information processing
        print("\n3. Testing MCS information processing...")
        
        # Test with SMARTS string
        mcs_smarts = "[#6](-[#6])-[#8]"  # Simple pattern
        mcs_mol = create_mcs_molecule_from_info(mcs_smarts)
        
        if mcs_mol:
            print(f"✅ Successfully created MCS molecule from SMARTS: {mcs_smarts}")
        else:
            print("❌ Failed to create MCS molecule from SMARTS")
        
        # Test with dictionary format
        mcs_info_dict = {
            "smarts": mcs_smarts,
            "atom_count": 3,
            "valid_smarts": True
        }
        
        mcs_mol_dict = create_mcs_molecule_from_info(mcs_info_dict)
        if mcs_mol_dict:
            print(f"✅ Successfully created MCS molecule from dictionary")
        else:
            print("❌ Failed to create MCS molecule from dictionary")
        
        # Test 4: Template info storage
        print("\n4. Testing template info storage...")
        template_info = {
            "name": "2hyy",
            "index": 0,
            "total_templates": 1,
            "template_pdb": "2hyy",
            "mcs_smarts": mcs_smarts,
            "atoms_matched": 3
        }
        
        session.set(SESSION_KEYS["TEMPLATE_INFO"], template_info)
        retrieved_info = session.get(SESSION_KEYS["TEMPLATE_INFO"])
        
        if retrieved_info and retrieved_info.get("mcs_smarts") == mcs_smarts:
            print("✅ Template info stored and retrieved correctly")
        else:
            print("❌ Template info storage failed")
            return False
        
        print("\n🎉 All visualization fix tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure RDKit and all required dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_manager_integration():
    """Test memory manager integration"""
    try:
        from rdkit import Chem
        from templ_pipeline.ui.core.memory_manager import get_memory_manager
        
        print("\n5. Testing memory manager integration...")
        
        memory_manager = get_memory_manager()
        
        # Test molecule storage
        test_mol = Chem.MolFromSmiles("CCO")
        success = memory_manager.store_molecule("test_mol", test_mol)
        
        if success:
            print("✅ Molecule stored in memory manager")
        else:
            print("❌ Failed to store molecule in memory manager")
            return False
        
        # Test molecule retrieval
        retrieved = memory_manager.get_molecule("test_mol")
        if retrieved and retrieved.GetNumAtoms() == test_mol.GetNumAtoms():
            print("✅ Molecule retrieved from memory manager")
        else:
            print("❌ Failed to retrieve molecule from memory manager")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")
        return False


if __name__ == "__main__":
    print("TEMPL Pipeline Visualization Fix Test")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_molecule_storage_and_retrieval()
    test2_passed = test_memory_manager_integration()
    
    if test1_passed and test2_passed:
        print("\n🎉 All tests passed! Visualization fixes are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1) 