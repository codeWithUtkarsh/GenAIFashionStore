#!/usr/bin/env python3
"""
Safe runner script for GenAI Fashion Store.
Handles common issues and provides a safer startup experience.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import argparse
import signal
import time

class SafeRunner:
    """Safe runner for the Fashion Store application."""

    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.system = platform.system()
        self.machine = platform.machine()

    def print_banner(self):
        """Print startup banner."""
        print("=" * 60)
        print("🛍️  GenAI Fashion Store - Safe Startup")
        print("=" * 60)
        print(f"System: {self.system} ({self.machine})")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Working Directory: {self.base_dir}")
        print("=" * 60)

    def check_python_version(self):
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True

    def set_environment_variables(self):
        """Set environment variables to prevent common issues."""
        print("\nSetting environment variables...")

        # Prevent threading issues
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        # Prevent memory issues
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        # macOS specific settings
        if self.system == 'Darwin':
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            if self.machine == 'arm64':
                # Apple Silicon specific
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                print("  ✅ Applied Apple Silicon optimizations")
            print("  ✅ Applied macOS settings")

        print("  ✅ Environment variables configured")

    def check_virtual_environment(self):
        """Check if running in virtual environment."""
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )

        if in_venv:
            print(f"✅ Virtual environment active: {sys.prefix}")
            return True
        else:
            print("⚠️  Not running in virtual environment")
            print("   Recommended: python3 -m venv .venv && source .venv/bin/activate")
            return True  # Continue anyway

    def check_env_file(self):
        """Check for .env configuration file."""
        env_file = self.base_dir / '.env'
        env_example = self.base_dir / '.env.example'

        if env_file.exists():
            print("✅ .env file found")
            return True
        elif env_example.exists():
            print("⚠️  .env file not found, creating from template...")
            shutil.copy2(env_example, env_file)
            print("   Please edit .env with your API keys")
            return True
        else:
            print("⚠️  No .env file found")
            return True  # Continue anyway

    def create_directories(self):
        """Create necessary directories."""
        print("\nCreating necessary directories...")

        directories = [
            'data', 'data/images', 'models', 'vector_db',
            'chroma_db', '.cache', '.cache/embeddings'
        ]

        for dir_name in directories:
            dir_path = self.base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)

        print("  ✅ Directories ready")

    def clear_cache(self, clear_all=False):
        """Clear Python and optionally Streamlit cache."""
        print("\nClearing cache...")

        # Clear Python cache
        for pattern in ['__pycache__', '*.pyc', '*.pyo']:
            for path in self.base_dir.rglob(pattern):
                try:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                except Exception:
                    pass

        print("  ✅ Python cache cleared")

        # Clear Streamlit cache if requested
        if clear_all:
            streamlit_cache_dirs = [
                Path.home() / '.streamlit' / 'cache',
                self.base_dir / '.streamlit' / 'cache'
            ]

            for cache_dir in streamlit_cache_dirs:
                if cache_dir.exists():
                    try:
                        shutil.rmtree(cache_dir)
                        print(f"  ✅ Cleared {cache_dir}")
                    except Exception:
                        pass

    def configure_streamlit(self):
        """Create Streamlit configuration for stability."""
        print("\nConfiguring Streamlit...")

        config_dir = Path.home() / '.streamlit'
        config_dir.mkdir(exist_ok=True)

        config_file = config_dir / 'config.toml'

        config_content = """
[server]
maxUploadSize = 50
maxMessageSize = 50
enableXsrfProtection = true
enableCORS = false

[runner]
magicEnabled = false
fastReruns = false

[browser]
gatherUsageStats = false

[logger]
level = "info"

[client]
showErrorDetails = false
"""

        with open(config_file, 'w') as f:
            f.write(config_content.strip())

        print("  ✅ Streamlit configured for stability")

    def check_dependencies(self):
        """Check if required packages are installed."""
        print("\nChecking dependencies...")

        required_packages = ['streamlit', 'torch', 'transformers', 'chromadb', 'PIL']
        missing = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)

        if missing:
            print(f"  ⚠️  Missing packages: {', '.join(missing)}")

            # Check for requirements file
            req_files = ['requirements_stable.txt', 'requirements.txt']
            req_file = None

            for rf in req_files:
                if (self.base_dir / rf).exists():
                    req_file = rf
                    break

            if req_file:
                response = input(f"\nInstall from {req_file}? (y/n): ")
                if response.lower() == 'y':
                    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', req_file])
                    print("  ✅ Dependencies installed")
                else:
                    print("  ⚠️  Continuing without installing dependencies")
            else:
                print("  ❌ No requirements file found")
                return False
        else:
            print("  ✅ All core dependencies installed")

        return True

    def run_streamlit(self, port=8501, debug=False):
        """Run Streamlit application with safety measures."""
        print(f"\n{'='*60}")
        print(f"Starting Streamlit on port {port}...")
        print(f"{'='*60}")
        print("\nAccess the application at:")
        print(f"  → http://localhost:{port}")
        print("\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")

        # Build Streamlit command
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', str(port),
            '--server.maxUploadSize', '50',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ]

        if debug:
            cmd.extend(['--logger.level', 'debug'])

        # Handle signals properly
        def signal_handler(sig, frame):
            print("\n\nShutting down gracefully...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        try:
            # Run Streamlit
            process = subprocess.Popen(cmd)
            process.wait()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            process.terminate()
            time.sleep(1)
            if process.poll() is None:
                process.kill()
        except Exception as e:
            print(f"\n❌ Error running Streamlit: {e}")
            return False

        return True

    def run(self, args):
        """Main run method."""
        self.print_banner()

        # Run checks
        if not self.check_python_version():
            return False

        self.set_environment_variables()
        self.check_virtual_environment()
        self.check_env_file()
        self.create_directories()

        if args.clear_cache:
            self.clear_cache(clear_all=True)
        else:
            self.clear_cache(clear_all=False)

        self.configure_streamlit()

        if not self.check_dependencies():
            if not args.force:
                print("\n❌ Dependencies check failed. Use --force to continue anyway.")
                return False

        # Run application
        return self.run_streamlit(port=args.port, debug=args.debug)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Safe runner for GenAI Fashion Store'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run Streamlit on (default: 8501)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear all caches before starting'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force run even with missing dependencies'
    )

    args = parser.parse_args()

    runner = SafeRunner()
    success = runner.run(args)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
