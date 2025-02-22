import os

# Define directory and file structure
structure = {
    'backend': {
        'config': {
            'metrics.yaml': '',
            'datasets/': {
                'COMPAS.csv': '',
                'GermanCredit.csv': ''
            }
        },
        'main.py': '',
        'bias_detector.py': '',
        'explainability.py': '',
        'mitigations.py': '',
        'db_manager.py': '',
        'requirements.txt': 'fastapi==0.95.0\nuvicorn==0.22.0\npandas==1.5.3\nfairlearn==0.9.0\nshap==0.41.0\nlime==0.2.0.1',
        'tests': {
            'test_bias.py': '',
            'test_mitigations.py': ''
        }
    },
    'frontend': {
        'public': {
            'index.html': '',
            'assets': {
                'styles.css': '',
                'viz.js': ''
            }
        },
        'src': {
            'app.py': '',
            'components': {
                'metric_card.py': '',
                'dataset_upload.py': ''
            },
            'pages': {
                'analyze.py': '',
                'reports.py': ''
            },
            'package.json': '',
            'README.md': ''
        }
    },
    '.gitignore': '',
    'LICENSE': 'MIT License',
    'setup.sh': '#!/bin/bash\n# Setup script to install dependencies\npip install -r requirements.txt\n',
    'README.md': '# AI Ethics Auditor\nA FOSS toolkit to detect, explain, and mitigate bias in AI/ML models and datasets.',
}

# Function to create files and directories
def create_structure(base_path, structure):
    for name, value in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(value, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, value)
        else:
            with open(path, 'w') as file:
                file.write(value)
            print(f'Created: {path}')

# Create the project structure
if __name__ == "__main__":
    project_root = 'ai-ethics-auditor'
    os.makedirs(project_root, exist_ok=True)
    create_structure(project_root, structure)
    print(f"Project structure created successfully in {project_root}!")
