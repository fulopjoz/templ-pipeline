# RNA-Protein Classifier

A standalone web-based tool for classifying molecules as RNA binders or Protein binders using binary classification and Graph Neural Networks.

## Overview

The RNA-Protein Classifier is a modern, interactive web application that provides:

- **Binary Classification**: Classify molecules as RNA binders or Protein binders
- **Multiple Input Methods**: Support for various molecular identifiers (SMILES, InChI, CID, etc.)
- **Database Integration**: Compatible with PubChem, ChEMBL, DrugBank, and custom datasets
- **Interactive Visualizations**: Real-time charts and detailed results tables
- **Modern UI**: Built with Tailwind CSS and animated with Vanta.js

## Features

### Input Options

- **Database Sources**: PubChem, ChEMBL, DrugBank, Custom Dataset
- **Identifier Types**: 
  - Compound CID
  - SMILES notation
  - InChI strings
  - Molecule names
- **Batch Processing**: Multiple molecules at once
- **File Upload**: Upload datasets directly
- **Sample Data**: Pre-loaded examples for testing

### Classification Results

- **Summary Statistics**: Total molecules, RNA binders, Protein binders, model accuracy
- **Visual Analytics**: Interactive doughnut charts showing distribution
- **Detailed Predictions**: Table with molecule ID, prediction, probability, and confidence scores
- **Export Capabilities**: Results can be viewed in both chart and table formats

## Usage

### Opening the Application

1. Open `rna_protein_classifier.html` in a modern web browser
2. The application will load with sample data automatically

### Classifying Molecules

1. **Select Database Source**: Choose from PubChem, ChEMBL, DrugBank, or Custom Dataset
2. **Choose Identifier Type**: Select the type of molecular identifier you're using
3. **Input Data**: 
   - Type or paste molecule identifiers (one per line)
   - Or click "Upload dataset file" to load from a file
   - Or click "Load sample data" to use pre-configured examples
4. **Optional**: Check "Apply preprocessing" to enable data preprocessing
5. **Click "Classify Molecules"**: The system will process your input and display results

### Interpreting Results

The results section shows:

- **Total Molecules**: Number of molecules analyzed
- **RNA Binders**: Count of molecules predicted to bind RNA
- **Protein Binders**: Count of molecules predicted to bind proteins
- **Model Accuracy**: Overall accuracy of the classification model
- **Distribution Chart**: Visual representation of the classification split
- **Detailed Table**: Individual predictions with probability and confidence scores

## Technical Details

### Technologies Used

- **Frontend Framework**: Vanilla JavaScript with modern ES6+
- **Styling**: Tailwind CSS for responsive design
- **Icons**: Feather Icons for UI elements
- **Charts**: Chart.js for data visualization
- **Animation**: Vanta.js for background effects

### Requirements

- Modern web browser with JavaScript enabled
- Internet connection (for CDN resources)

### Demo Mode

The current implementation includes a demonstration mode that generates mock results for testing the UI. This allows you to:

- Test the interface without a backend
- Verify the visualization components
- Understand the expected data format

## Sample Data

The application includes sample SMILES strings for testing:

```
CC1=C(C(=CC=C1)C)NC(=O)CN(C)C
CC(C)NCC(C1=CC(=C(C=C1)O)OC)O
CN1C(=O)C(O)C(NC2=NC=CC(=N2)C(F)(F)F)C1=O
CC(C)(C)NC(=O)C1=CC(=C(C=C1)Cl)Cl
CC1=CC=C(C=C1)C(=O)NC2=CC=CC=C2
CN(C)C(=O)OC1=CC=CC=C1
CC(C)C1=CC(=CC(=C1)C(C)C)C(=O)O
CC1=CC(=CC=C1NC(=O)C)C
C1=CC(=CC=C1C(=O)O)N
CC(C)(C)C(=O)NC1=CC=C(C=C1)C#N
```

## Integration with TEMPL Pipeline

This classifier is part of the TEMPL Pipeline ecosystem. While the main TEMPL Pipeline focuses on template-based protein-ligand pose prediction, this RNA-Protein Classifier provides complementary functionality for molecular binding classification.

## Research Background

Based on research from [Fülöp József's thesis](https://github.com/fulopjoz/diploma_thesis), this classifier leverages Graph Neural Networks for binary classification of molecular binding targets.

## License

This tool is distributed under the same license as the TEMPL Pipeline:
- Software Code: MIT License
- Documentation: CC BY 4.0

## Support

For questions, issues, or contributions:
- GitHub Issues: [TEMPL Pipeline Issues](https://github.com/fulopjoz/templ-pipeline/issues)
- GitHub Discussions: [TEMPL Pipeline Discussions](https://github.com/fulopjoz/templ-pipeline/discussions)
