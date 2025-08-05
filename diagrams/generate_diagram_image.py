#!/usr/bin/env python3
"""
Generate Protein Superimposition Workflow Diagram as Image
"""

import subprocess
import sys
import os

def generate_mermaid_diagram():
    """Generate the Mermaid diagram as an image."""
    
    # Mermaid diagram code
    mermaid_code = """
flowchart TD
    Start([transform_ligand]) --> Load[Load Protein Structures]
    Load --> Filter[Filter Amino Acids]
    Filter --> Validate[Validate & Select Chains]
    Validate --> CheckAtoms{Enough CA Atoms?}
    
    CheckAtoms -->|No| Fail[Return None]
    CheckAtoms -->|Yes| Level1[Level 1: Homologous Alignment]
    
    Level1 --> CheckLength{Sequence Length Similar?}
    CheckLength -->|No| Level2[Level 2: Sequence Alignment]
    CheckLength -->|Yes| TryHomologs[Try superimpose_homologs]
    
    TryHomologs --> CheckAnchors{Sufficient Anchors?}
    CheckAnchors -->|No| Level2
    CheckAnchors -->|Yes| CalcRMSD[Calculate CA RMSD]
    
    Level2 --> CheckSeq{Sequence Alignment Success?}
    CheckSeq -->|No| Level3[Level 3: 3Di Structural]
    CheckSeq -->|Yes| CalcRMSD
    
    Level3 --> Check3Di{3Di Available & Success?}
    Check3Di -->|No| Level4[Level 4: Centroid Fallback]
    Check3Di -->|Yes| CalcRMSD
    
    Level4 --> CheckMin{Min Atoms Available?}
    CheckMin -->|No| Fail
    CheckMin -->|Yes| SimpleAlign[Simple Superimposition]
    SimpleAlign --> CalcRMSD
    
    CalcRMSD --> CheckThreshold{CA RMSD <= Threshold?}
    CheckThreshold -->|No| Fallback[Apply Fallback Thresholds]
    CheckThreshold -->|Yes| Transform[Transform Ligand Coordinates]
    
    Fallback --> CheckTemplates{Any Templates Pass?}
    CheckTemplates -->|No| BestTemplate[Use Best Available Template]
    CheckTemplates -->|Yes| Transform
    
    Transform --> ApplyMatrix[Apply Transformation Matrix]
    ApplyMatrix --> ValidateBonds[Validate Bond Lengths]
    ValidateBonds --> AddMetadata[Add Metadata Properties]
    AddMetadata --> Success[Return Transformed Ligand]
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef error fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef success fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    
    class Start,Success startEnd
    class Load,Filter,Validate,Level1,Level2,Level3,Level4,TryHomologs,CalcRMSD,Transform,ApplyMatrix,ValidateBonds,AddMetadata,SimpleAlign process
    class CheckAtoms,CheckLength,CheckAnchors,CheckSeq,Check3Di,CheckMin,CheckThreshold,CheckTemplates decision
    class Fail error
    class Fallback,BestTemplate process
"""
    
    # Create HTML file with the diagram
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Protein Superimposition Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: white;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{mermaid_code}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open('diagram.html', 'w') as f:
        f.write(html_content)
    
    print("HTML file 'diagram.html' created successfully!")
    print("You can open this file in a web browser to view the diagram.")
    print("Or use a tool like wkhtmltopdf to convert it to an image.")
    
    # Try to open in browser if possible
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.abspath('diagram.html'))
        print("Opened diagram in browser!")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open 'diagram.html' manually in your browser.")

def print_diagram_info():
    """Print information about the diagram and functions."""
    
    print("=" * 80)
    print("PROTEIN SUPERIMPOSITION WORKFLOW DIAGRAM")
    print("=" * 80)
    
    print("\nKey Functions Used in Protein Superimposition:")
    print("\nMain Functions:")
    functions = [
        "transform_ligand() - Primary orchestrator function",
        "superimpose_homologs() - Level 1: homologous alignment", 
        "superimpose() - Basic superimposition (used in fallbacks)",
        "_align_with_biotite_sequence() - Level 2: sequence-based alignment",
        "_align_with_3di_structural() - Level 3: 3Di structural alignment",
        "_align_with_centroid_fallback() - Level 4: centroid-based fallback"
    ]
    
    for i, func in enumerate(functions, 1):
        print(f"{i}. {func}")
    
    print("\nHelper Functions:")
    helper_functions = [
        "_rmsd_from_alignment() - Extract anchors and perform superimposition",
        "_validate_and_select_chains() - Validate chain selection",
        "filter_amino_acids() - Filter protein structures",
        "get_chains() - Extract chain information", 
        "to_sequence() - Convert structures to sequences",
        "align_optimal() - Perform optimal alignment",
        "rmsd() - Calculate RMSD"
    ]
    
    for i, func in enumerate(helper_functions, 1):
        print(f"{i}. {func}")
    
    print("\nWorkflow Features:")
    print("\nMulti-Level Fallback Strategy:")
    levels = [
        "Level 1: Homologous alignment for similar-length sequences",
        "Level 2: Sequence-based optimal alignment with anchor extraction", 
        "Level 3: 3Di structural alphabet alignment for remote homologs",
        "Level 4: Simple centroid-based alignment as final fallback"
    ]
    
    for level in levels:
        print(f"• {level}")
    
    print("\nQuality Control:")
    quality_controls = [
        "CA RMSD Thresholds: Progressive fallback thresholds (10Å, 15Å, 20Å)",
        "Anchor Count Validation: Minimum 3 anchors required",
        "Bond Length Validation: Ensures reasonable molecular geometry",
        "Chain Selection: Validates binding site chains from embedding data"
    ]
    
    for control in quality_controls:
        print(f"• {control}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_diagram_info()
    generate_mermaid_diagram() 