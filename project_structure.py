import os

# List of folders to create in the current repo
folders = [
    "config",
    "data/raw",
    "data/processed",
    "artifacts/models",
    "artifacts/transformers",
    "artifacts/metrics",
    "logs",
    "src",
    "src/components",
    "src/entity",
    "src/pipeline",
    "src/utils",
    "notebooks",
    "tests",
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Add __init__.py in every Python module folder
src_subfolders = [
    "src",
    "src/components",
    "src/entity",
    "src/pipeline",
    "src/utils",
]

for sf in src_subfolders:
    with open(f"{sf}/__init__.py", "w") as f:
        pass

# List of files to create
files = {
    "README.md": "",
    "requirements.txt": "",
    "setup.py": "",
    "main.py": "",
    "app.py": "",
    ".gitignore": "",

    # Config
    "config/config.yaml": "",
    "config/configuration.py": "",

    # Logs
    "logs/AICheatTextGuard.log": "",

    # Components
    "src/components/data_ingestion.py": "",
    "src/components/data_transformation.py": "",
    "src/components/model_trainer.py": "",
    "src/components/model_evaluation.py": "",

    # Entities
    "src/entity/config_entity.py": "",
    "src/entity/artifact_entity.py": "",

    # Pipelines
    "src/pipeline/training_pipeline.py": "",
    "src/pipeline/prediction_pipeline.py": "",

    # Utils
    "src/utils/common.py": "",
    "src/utils/logger.py": "",

    # Tests
    "tests/test_basic.py": "",
}

# Create files
for file_path, content in files.items():
    with open(file_path, "w") as f:
        f.write(content)

print("Project structure created successfully in this repo!")
