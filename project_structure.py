import os


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[+] Created folder: {path}")
    else:
        print(f"[=] Folder already exists: {path}")


def create_file(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            pass
        print(f"[+] Created file: {path}")
    else:
        print(f"[=] File already exists: {path}")


# List of folders to create
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

# Create folders safely
for folder in folders:
    create_folder(folder)

# Python module folders -> ensure __init__.py exists
src_subfolders = [
    "src",
    "src/components",
    "src/entity",
    "src/pipeline",
    "src/utils",
]

for sf in src_subfolders:
    create_file(f"{sf}/__init__.py")

# Files to create safely
files = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "main.py",
    "app.py",
    ".gitignore",
    # Config
    "config/config.yaml",
    "config/configuration.py",
    # Logs
    "logs/AICheatTextGuard.log",
    # Components
    "src/components/data_ingestion.py",
    "src/components/data_transformation.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    # Entity
    "src/entity/config_entity.py",
    "src/entity/artifact_entity.py",
    # Pipeline
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    # Utils
    "src/utils/common.py",
    "src/utils/logger.py",
    "src/utils/exception.py",
    # Tests
    "tests/test_basic.py",
]

# Create files safely
for file_path in files:
    create_file(file_path)

print("\n🎉 SAFE Project structure check complete!")
