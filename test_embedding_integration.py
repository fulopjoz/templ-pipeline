#!/usr/bin/env python3
"""
Test script to verify embedding functionality works with new dependencies
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        print(f"   Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        from transformers import EsmModel, EsmTokenizer
        print("✅ ESM components imported successfully")
    except ImportError as e:
        print(f"❌ ESM components import failed: {e}")
        return False
    
    return True

def test_esm_model():
    """Test ESM2 model loading"""
    print("\nTesting ESM2 model loading...")
    
    try:
        from transformers import EsmModel, EsmTokenizer
        import torch
        
        model_id = "facebook/esm2_t33_650M_UR50D"
        print(f"Loading {model_id}...")
        
        tokenizer = EsmTokenizer.from_pretrained(model_id)
        model = EsmModel.from_pretrained(model_id)
        
        print("✅ ESM2 model loaded successfully")
        print(f"   Model parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        return True
    except Exception as e:
        print(f"❌ ESM2 model loading failed: {e}")
        return False

def test_embedding_generation():
    """Test embedding generation with a simple sequence"""
    print("\nTesting embedding generation...")
    
    try:
        # Add the templ_pipeline to path
        sys.path.insert(0, '.')
        
        from templ_pipeline.core.embedding import calculate_embedding
        
        # Test with a simple protein sequence
        test_sequence = "MKQLEDKVEELLSKNYHLENEVARLKKLVGER"
        print(f"Test sequence: {test_sequence}")
        
        embedding = calculate_embedding(test_sequence)
        
        if embedding is not None:
            print(f"✅ Embedding generated successfully")
            print(f"   Shape: {embedding.shape}")
            print(f"   Mean: {embedding.mean():.6f}")
            print(f"   Std: {embedding.std():.6f}")
            return True
        else:
            print("❌ Embedding generation returned None")
            return False
            
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_manager():
    """Test EmbeddingManager functionality"""
    print("\nTesting EmbeddingManager...")
    
    try:
        from templ_pipeline.core.embedding import EmbeddingManager
        
        # Try to initialize with default path
        manager = EmbeddingManager()
        print("✅ EmbeddingManager initialized successfully")
        
        # Check if any embeddings are loaded
        if hasattr(manager, 'embedding_db') and manager.embedding_db:
            print(f"   Pre-computed embeddings: {len(manager.embedding_db)}")
        else:
            print("   No pre-computed embeddings loaded (this is OK)")
        
        return True
        
    except Exception as e:
        print(f"❌ EmbeddingManager test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing TEMPL Embedding Integration")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("ESM2 Model Test", test_esm_model),
        ("Embedding Generation Test", test_embedding_generation),
        ("EmbeddingManager Test", test_embedding_manager)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Embedding functionality is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
