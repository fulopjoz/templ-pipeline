#!/usr/bin/env python3
"""
Enhanced Protein Superimposition Workflow Diagram Generator
With detailed descriptions of each stage and function usage
"""

import os

def generate_enhanced_diagram():
    """Generate enhanced Mermaid diagram with detailed descriptions."""
    
    # Enhanced Mermaid diagram with detailed descriptions
    mermaid_code = """
flowchart TD
    Start([transform_ligand<br/>Primary Orchestrator]) --> Load[Load Protein Structures<br/>load_reference_protein()<br/>load_target_data()]
    Load --> Filter[Filter Amino Acids<br/>filter_amino_acids()<br/>Extract protein residues only]
    Filter --> Validate[Validate & Select Chains<br/>_validate_and_select_chains()<br/>get_chains()]
    Validate --> CheckAtoms{Enough CA Atoms?<br/>MIN_CA_ATOMS_FOR_ALIGNMENT = 3}
    
    CheckAtoms -->|No| Fail[Return None<br/>Insufficient atoms for alignment]
    CheckAtoms -->|Yes| Level1[Level 1: Homologous Alignment<br/>For similar-length sequences]
    
    Level1 --> CheckLength{Sequence Length Similar?<br/>Length difference < 30%}
    CheckLength -->|No| Level2[Level 2: Sequence Alignment<br/>_align_with_biotite_sequence()<br/>to_sequence() + align_optimal()]
    CheckLength -->|Yes| TryHomologs[Try superimpose_homologs()<br/>BLOSUM62 matrix<br/>Gap penalty: -10]
    
    TryHomologs --> CheckAnchors{Sufficient Anchors?<br/>MIN_ANCHOR_RESIDUES = 15}
    CheckAnchors -->|No| Level2
    CheckAnchors -->|Yes| CalcRMSD[Calculate CA RMSD<br/>rmsd() function<br/>Measure alignment quality]
    
    Level2 --> CheckSeq{Sequence Alignment Success?<br/>BLOSUM62 + optimal alignment}
    CheckSeq -->|No| Level3[Level 3: 3Di Structural<br/>_align_with_3di_structural()<br/>3Di structural alphabet]
    CheckSeq -->|Yes| CalcRMSD
    
    Level3 --> Check3Di{3Di Available & Success?<br/>STRUCTURAL_ALPHABET_AVAILABLE}
    Check3Di -->|No| Level4[Level 4: Centroid Fallback<br/>_align_with_centroid_fallback()<br/>Simple superimpose()]
    Check3Di -->|Yes| CalcRMSD
    
    Level4 --> CheckMin{Min Atoms Available?<br/>MIN_CA_ATOMS_FOR_ALIGNMENT}
    CheckMin -->|No| Fail
    CheckMin -->|Yes| SimpleAlign[Simple Superimposition<br/>superimpose()<br/>First N atoms only]
    SimpleAlign --> CalcRMSD
    
    CalcRMSD --> CheckThreshold{CA RMSD <= Threshold?<br/>CA_RMSD_THRESHOLD = 10.0Å}
    CheckThreshold -->|No| Fallback[Apply Fallback Thresholds<br/>CA_RMSD_FALLBACK_THRESHOLDS<br/>[10.0, 15.0, 20.0]Å]
    CheckThreshold -->|Yes| Transform[Transform Ligand Coordinates<br/>Apply transformation matrix<br/>Point3D coordinates]
    
    Fallback --> CheckTemplates{Any Templates Pass?<br/>filter_templates_by_ca_rmsd()]
    CheckTemplates -->|No| BestTemplate[Use Best Available Template<br/>get_templates_with_progressive_fallback()]
    CheckTemplates -->|Yes| Transform
    
    Transform --> ApplyMatrix[Apply Transformation Matrix<br/>transformation.apply(coords)<br/>Bulk coordinate transformation]
    ApplyMatrix --> ValidateBonds[Validate Bond Lengths<br/>Check 0.5-3.0Å range<br/>Ensure molecular geometry]
    ValidateBonds --> AddMetadata[Add Metadata Properties<br/>ca_rmsd, template_pid<br/>alignment_method, anchor_count]
    AddMetadata --> Success[Return Transformed Ligand<br/>Enhanced with metadata<br/>Ready for downstream processing]
    
    %% Styling with enhanced colors
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef success fill:#e8f5e8,stroke:#2e7d32,stroke-width:3px
    classDef level1 fill:#e8faf5,stroke:#00695c,stroke-width:2px
    classDef level2 fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef level3 fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    classDef level4 fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class Start,Success startEnd
    class Load,Filter,Validate,ApplyMatrix,ValidateBonds,AddMetadata process
    class CheckAtoms,CheckLength,CheckAnchors,CheckSeq,Check3Di,CheckMin,CheckThreshold,CheckTemplates decision
    class Fail error
    class Fallback,BestTemplate process
    class TryHomologs,CalcRMSD level1
    class Level2 level2
    class Level3 level3
    class Level4,SimpleAlign level4
    class Transform process
"""
    
    # Create enhanced HTML file
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Protein Superimposition Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .mermaid {{
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
        }}
        .description {{
            margin-top: 40px;
            padding: 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .description h3 {{
            color: white;
            margin-top: 0;
        }}
        .function-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .function-card {{
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
        }}
        .function-card h4 {{
            color: #ffd700;
            margin-top: 0;
        }}
        .function-card ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .function-card li {{
            margin: 5px 0;
        }}
        .workflow-features {{
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }}
        .color-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
            justify-content: center;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #333;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Enhanced Protein Superimposition Workflow</h1>
        
        <div class="mermaid">
{mermaid_code}
        </div>
        
        <div class="description">
            <h3>🔍 Detailed Function Usage & Stage Descriptions</h3>
            
            <div class="function-grid">
                <div class="function-card">
                    <h4>📥 Data Loading Stage</h4>
                    <ul>
                        <li><strong>load_reference_protein()</strong>: Loads reference PDB structure using biotite</li>
                        <li><strong>load_target_data()</strong>: Creates target molecule from SMILES and adds hydrogens</li>
                        <li><strong>filter_amino_acids()</strong>: Extracts only protein residues for alignment</li>
                        <li><strong>get_chains()</strong>: Identifies available protein chains</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>🔗 Chain Validation Stage</h4>
                    <ul>
                        <li><strong>_validate_and_select_chains()</strong>: Validates binding site chains from embedding data</li>
                        <li>Falls back to all available chains if specified chains not found</li>
                        <li>Ensures sufficient CA atoms for alignment (minimum 3)</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>🎯 Level 1: Homologous Alignment</h4>
                    <ul>
                        <li><strong>superimpose_homologs()</strong>: Uses BLOSUM62 substitution matrix</li>
                        <li>Requires sequence length similarity (< 30% difference)</li>
                        <li>Minimum 15 anchor residues for quality alignment</li>
                        <li>Gap penalty: -10, terminal penalty: True</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>🧬 Level 2: Sequence Alignment</h4>
                    <ul>
                        <li><strong>_align_with_biotite_sequence()</strong>: Optimal sequence alignment</li>
                        <li><strong>to_sequence()</strong>: Converts protein structures to sequences</li>
                        <li><strong>align_optimal()</strong>: Performs optimal alignment with BLOSUM62</li>
                        <li><strong>_rmsd_from_alignment()</strong>: Extracts anchors and calculates RMSD</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>🏗️ Level 3: 3Di Structural Alignment</h4>
                    <ul>
                        <li><strong>_align_with_3di_structural()</strong>: Uses 3Di structural alphabet</li>
                        <li>Requires STRUCTURAL_ALPHABET_AVAILABLE = True</li>
                        <li>Specialized for remote homologs with poor sequence similarity</li>
                        <li>Uses 3Di substitution matrix for structural similarity</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>⚡ Level 4: Centroid Fallback</h4>
                    <ul>
                        <li><strong>_align_with_centroid_fallback()</strong>: Simple centroid-based alignment</li>
                        <li><strong>superimpose()</strong>: Basic structural superimposition</li>
                        <li>Uses first N atoms from both structures (minimum 3)</li>
                        <li>Final fallback when all other methods fail</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>📊 Quality Assessment</h4>
                    <ul>
                        <li><strong>rmsd()</strong>: Calculates CA RMSD between structures</li>
                        <li><strong>filter_templates_by_ca_rmsd()</strong>: Filters by RMSD threshold</li>
                        <li>Progressive fallback thresholds: 10Å, 15Å, 20Å</li>
                        <li>Bond length validation: 0.5-3.0Å range</li>
                    </ul>
                </div>
                
                <div class="function-card">
                    <h4>🔄 Coordinate Transformation</h4>
                    <ul>
                        <li><strong>transformation.apply()</strong>: Bulk coordinate transformation</li>
                        <li>Point3D coordinate setting for each atom</li>
                        <li>Bond length validation ensures molecular geometry</li>
                        <li>Metadata addition: ca_rmsd, template_pid, alignment_method</li>
                    </ul>
                </div>
            </div>
            
            <div class="workflow-features">
                <h4>🎯 Workflow Features & Quality Control</h4>
                <ul>
                    <li><strong>Multi-Level Fallback Strategy</strong>: 4 progressive alignment methods for maximum robustness</li>
                    <li><strong>CA RMSD Thresholds</strong>: Progressive fallback (10Å → 15Å → 20Å) with central atom final fallback</li>
                    <li><strong>Anchor Count Validation</strong>: Minimum 3 anchors required for reliable alignment</li>
                    <li><strong>Bond Length Validation</strong>: Ensures reasonable molecular geometry (0.5-3.0Å range)</li>
                    <li><strong>Chain Selection</strong>: Validates binding site chains from embedding data with fallback to all chains</li>
                    <li><strong>Metadata Tracking</strong>: Records alignment method, RMSD, anchor count, and chain information</li>
                </ul>
            </div>
            
            <div class="color-legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e1f5fe;"></div>
                    <span>Start/End Points</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f3e5f5;"></div>
                    <span>Process Steps</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fff3e0;"></div>
                    <span>Decision Points</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e8faf5;"></div>
                    <span>Level 1: Homologous</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f3e5f5;"></div>
                    <span>Level 2: Sequence</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fff8e1;"></div>
                    <span>Level 3: 3Di Structural</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #fce4ec;"></div>
                    <span>Level 4: Centroid</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #ffebee;"></div>
                    <span>Error/Failure</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e8f5e8;"></div>
                    <span>Success</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis',
                nodeSpacing: 50,
                rankSpacing: 50
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write enhanced HTML file
    with open('enhanced_diagram.html', 'w') as f:
        f.write(html_content)
    
    print("✅ Enhanced HTML file 'enhanced_diagram.html' created successfully!")
    print("📊 Features enhanced diagram with:")
    print("   • Detailed function descriptions")
    print("   • Stage-by-stage explanations")
    print("   • Color-coded workflow levels")
    print("   • Searchable function references")
    print("   • Quality control explanations")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath('enhanced_diagram.html'))
        print("🌐 Opened enhanced diagram in browser!")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"📁 Please open 'enhanced_diagram.html' manually in your browser.")

def print_searchable_functions():
    """Print searchable function reference for easy lookup."""
    
    print("\n" + "="*100)
    print("🔍 SEARCHABLE FUNCTION REFERENCE")
    print("="*100)
    
    functions = {
        "Data Loading": [
            "load_reference_protein() - Load reference PDB structure",
            "load_target_data() - Create target molecule from SMILES",
            "filter_amino_acids() - Extract protein residues only",
            "get_chains() - Get available protein chains"
        ],
        "Chain Validation": [
            "_validate_and_select_chains() - Validate binding site chains",
            "MIN_CA_ATOMS_FOR_ALIGNMENT - Minimum atoms constant (3)"
        ],
        "Level 1 - Homologous": [
            "superimpose_homologs() - BLOSUM62 homologous alignment",
            "MIN_ANCHOR_RESIDUES - Minimum anchors constant (15)"
        ],
        "Level 2 - Sequence": [
            "_align_with_biotite_sequence() - Sequence-based alignment",
            "to_sequence() - Convert structures to sequences",
            "align_optimal() - Perform optimal alignment",
            "_rmsd_from_alignment() - Extract anchors and calculate RMSD"
        ],
        "Level 3 - 3Di Structural": [
            "_align_with_3di_structural() - 3Di structural alphabet",
            "STRUCTURAL_ALPHABET_AVAILABLE - 3Di availability check"
        ],
        "Level 4 - Centroid": [
            "_align_with_centroid_fallback() - Centroid-based fallback",
            "superimpose() - Basic structural superimposition"
        ],
        "Quality Assessment": [
            "rmsd() - Calculate CA RMSD",
            "filter_templates_by_ca_rmsd() - Filter by RMSD threshold",
            "CA_RMSD_THRESHOLD - Primary threshold (10Å)",
            "CA_RMSD_FALLBACK_THRESHOLDS - Fallback thresholds [10,15,20]Å"
        ],
        "Coordinate Transformation": [
            "transformation.apply() - Bulk coordinate transformation",
            "Point3D - RDKit coordinate setting",
            "validate_template_molecule() - Template validation",
            "enhance_template_with_metadata() - Add metadata properties"
        ]
    }
    
    for category, funcs in functions.items():
        print(f"\n📂 {category}:")
        for func in funcs:
            print(f"   • {func}")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    print_searchable_functions()
    generate_enhanced_diagram() 