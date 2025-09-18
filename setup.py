#!/usr/bin/env python3
"""
Setup script for initializing the GenAI Fashion Store application.
This script handles environment setup, dependency installation, and initial data loading.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json
import shutil
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FashionStoreSetup:
    """Setup manager for the Fashion Store application."""

    def __init__(self):
        """Initialize setup manager."""
        self.base_dir = Path(__file__).parent.absolute()
        self.venv_path = self.base_dir / ".venv"
        self.requirements_file = self.base_dir / "requirements.txt"
        self.env_file = self.base_dir / ".env"
        self.env_example_file = self.base_dir / ".env.example"

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        required_version = (3, 8)
        current_version = sys.version_info[:2]

        if current_version < required_version:
            logger.error(f"Python {required_version[0]}.{required_version[1]} or higher is required. "
                        f"Current version: {current_version[0]}.{current_version[1]}")
            return False

        logger.info(f"Python version {current_version[0]}.{current_version[1]} ✓")
        return True

    def create_virtual_environment(self) -> bool:
        """Create a Python virtual environment."""
        try:
            if self.venv_path.exists():
                logger.info("Virtual environment already exists")
                return True

            logger.info("Creating virtual environment...")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_path)],
                check=True
            )
            logger.info("Virtual environment created successfully ✓")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create virtual environment: {e}")
            return False

    def get_pip_command(self) -> List[str]:
        """Get the pip command for the virtual environment."""
        if os.name == 'nt':  # Windows
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            pip_path = self.venv_path / "bin" / "pip"

        if pip_path.exists():
            return [str(pip_path)]
        else:
            return [sys.executable, "-m", "pip"]

    def install_dependencies(self) -> bool:
        """Install required Python dependencies."""
        try:
            if not self.requirements_file.exists():
                logger.error(f"Requirements file not found: {self.requirements_file}")
                return False

            pip_cmd = self.get_pip_command()

            # Upgrade pip
            logger.info("Upgrading pip...")
            subprocess.run(
                pip_cmd + ["install", "--upgrade", "pip"],
                check=True,
                capture_output=True
            )

            # Install dependencies
            logger.info("Installing dependencies from requirements.txt...")
            subprocess.run(
                pip_cmd + ["install", "-r", str(self.requirements_file)],
                check=True
            )

            logger.info("Dependencies installed successfully ✓")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            logger.error("Please try installing manually with: pip install -r requirements.txt")
            return False

    def setup_environment_file(self) -> bool:
        """Setup the .env file from example."""
        try:
            if self.env_file.exists():
                logger.info(".env file already exists")
                response = input("Do you want to update your .env file? (y/n): ").lower()
                if response != 'y':
                    return True

            if not self.env_example_file.exists():
                logger.error(f"Environment example file not found: {self.env_example_file}")
                return False

            # Copy example file
            shutil.copy2(self.env_example_file, self.env_file)
            logger.info(f"Created .env file from example ✓")

            # Prompt for API keys
            logger.info("\n" + "="*50)
            logger.info("API KEY CONFIGURATION")
            logger.info("="*50)

            print("\nThe application requires the following API keys:")
            print("1. OpenAI API Key - For GPT-powered chat assistant")
            print("2. Kaggle API Key - For downloading the fashion dataset")
            print("\nYou can get these keys from:")
            print("- OpenAI: https://platform.openai.com/api-keys")
            print("- Kaggle: https://www.kaggle.com/settings/account")

            response = input("\nDo you want to configure API keys now? (y/n): ").lower()

            if response == 'y':
                # Read current .env content
                with open(self.env_file, 'r') as f:
                    env_content = f.read()

                # Get OpenAI API Key
                openai_key = input("Enter your OpenAI API Key (or press Enter to skip): ").strip()
                if openai_key:
                    env_content = env_content.replace(
                        "OPENAI_API_KEY=your_openai_api_key_here",
                        f"OPENAI_API_KEY={openai_key}"
                    )

                # Get Kaggle credentials
                kaggle_username = input("Enter your Kaggle Username (or press Enter to skip): ").strip()
                kaggle_key = input("Enter your Kaggle API Key (or press Enter to skip): ").strip()

                if kaggle_username and kaggle_key:
                    env_content = env_content.replace(
                        "KAGGLE_USERNAME=your_kaggle_username",
                        f"KAGGLE_USERNAME={kaggle_username}"
                    )
                    env_content = env_content.replace(
                        "KAGGLE_KEY=your_kaggle_api_key",
                        f"KAGGLE_KEY={kaggle_key}"
                    )

                # Save updated content
                with open(self.env_file, 'w') as f:
                    f.write(env_content)

                logger.info("API keys configured successfully ✓")
            else:
                logger.info("Skipping API key configuration. Please update .env file manually.")

            return True

        except Exception as e:
            logger.error(f"Failed to setup environment file: {e}")
            return False

    def create_directories(self) -> bool:
        """Create required directories."""
        try:
            directories = [
                "data",
                "data/images",
                "models",
                "vector_db",
                "chroma_db",
                ".cache",
                ".cache/embeddings"
            ]

            for dir_name in directories:
                dir_path = self.base_dir / dir_name
                dir_path.mkdir(parents=True, exist_ok=True)

            logger.info("Created required directories ✓")
            return True

        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False

    def download_sample_data(self) -> bool:
        """Download sample data if Kaggle credentials are configured."""
        try:
            # Check if Kaggle credentials are configured
            if not self.env_file.exists():
                logger.info("Skipping data download - .env file not configured")
                return True

            # Load environment variables
            from dotenv import load_dotenv
            load_dotenv(self.env_file)

            kaggle_username = os.getenv("KAGGLE_USERNAME")
            kaggle_key = os.getenv("KAGGLE_KEY")

            if not kaggle_username or not kaggle_key:
                logger.info("Skipping data download - Kaggle credentials not configured")
                logger.info("You can download the dataset later by running the application")
                return True

            # Try to download dataset
            logger.info("Attempting to download fashion dataset from Kaggle...")

            from data_downloader import FashionDatasetDownloader
            downloader = FashionDatasetDownloader(max_items=100)  # Small sample for setup
            success, _ = downloader.setup_dataset()

            if success:
                logger.info("Sample data downloaded successfully ✓")
            else:
                logger.warning("Could not download dataset. You can try again when running the application.")

            return True

        except ImportError:
            logger.info("Data downloader not available yet. Dataset will be downloaded on first run.")
            return True
        except Exception as e:
            logger.warning(f"Could not download sample data: {e}")
            logger.info("Dataset will be downloaded when you run the application")
            return True

    def print_instructions(self):
        """Print post-setup instructions."""
        print("\n" + "="*60)
        print("🎉 SETUP COMPLETE!")
        print("="*60)
        print("\nTo run the Fashion Store application:")
        print("\n1. Activate the virtual environment:")

        if os.name == 'nt':  # Windows
            print(f"   .venv\\Scripts\\activate")
        else:  # Unix-like
            print(f"   source .venv/bin/activate")

        print("\n2. Run the application:")
        print("   streamlit run app.py")

        print("\n3. Open your browser to:")
        print("   http://localhost:8501")

        print("\n" + "-"*60)
        print("📚 DOCUMENTATION")
        print("-"*60)
        print("\nFor more information, see:")
        print("- README.md - Project documentation")
        print("- config.py - Configuration settings")
        print("- .env - API key configuration")

        print("\n" + "-"*60)
        print("🔧 TROUBLESHOOTING")
        print("-"*60)
        print("\nIf you encounter issues:")
        print("1. Ensure all API keys are configured in .env")
        print("2. Check that port 8501 is available")
        print("3. Review logs in the terminal for error messages")
        print("4. Ensure you have at least 4GB of free disk space")

        print("\n" + "-"*60)
        print("💡 TIPS")
        print("-"*60)
        print("\n- The first run will download and index the dataset (may take a few minutes)")
        print("- Use a smaller dataset for faster development (adjust MAX_PRODUCTS_TO_LOAD in config.py)")
        print("- Enable GPU support for better performance (if available)")
        print("\n" + "="*60)

    def run_setup(self) -> bool:
        """Run the complete setup process."""
        print("\n" + "="*60)
        print("🚀 GenAI Fashion Store - Setup")
        print("="*60)

        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating virtual environment", self.create_virtual_environment),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up environment file", self.setup_environment_file),
            ("Creating directories", self.create_directories),
            ("Downloading sample data", self.download_sample_data)
        ]

        for step_name, step_func in steps:
            print(f"\n📌 {step_name}...")
            if not step_func():
                logger.error(f"Setup failed at: {step_name}")
                return False

        return True


def main():
    """Main setup function."""
    setup = FashionStoreSetup()

    try:
        if setup.run_setup():
            setup.print_instructions()
            sys.exit(0)
        else:
            logger.error("Setup failed. Please check the errors above.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
