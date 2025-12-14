"""Helper script to set up REFPROP_9 from HyFlux_Hx repository."""

import os
import sys
import subprocess
from pathlib import Path


def clone_hyflux_repo(target_dir=None):
    """
    Clone HyFlux_Hx repository to get REFPROP_9.
    
    Parameters
    ----------
    target_dir : str, optional
        Target directory. Defaults to ~/HyFlux_Hx
    """
    if target_dir is None:
        target_dir = os.path.join(os.path.expanduser('~'), 'HyFlux_Hx')
    
    repo_url = 'https://github.com/psperera/HyFlux_Hx.git'
    
    if os.path.exists(target_dir):
        print(f"Directory {target_dir} already exists.")
        print("Skipping clone. If you want to update, run:")
        print(f"  cd {target_dir} && git pull")
        return target_dir
    
    print(f"Cloning HyFlux_Hx repository to {target_dir}...")
    try:
        subprocess.run(['git', 'clone', repo_url, target_dir], check=True)
        print(f"✓ Repository cloned successfully")
        return target_dir
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone repository: {e}")
        return None
    except FileNotFoundError:
        print("✗ Git not found. Please install git or clone manually:")
        print(f"  git clone {repo_url} {target_dir}")
        return None


def create_symlink(source_dir, link_path):
    """Create symlink to REFPROP_9 directory."""
    refprop9_path = os.path.join(source_dir, 'Hyflux', 'REFPROP_9')
    
    if not os.path.exists(refprop9_path):
        print(f"✗ REFPROP_9 not found at {refprop9_path}")
        return False
    
    # Create symlink in project root
    project_root = Path(__file__).parent.parent
    symlink_target = project_root / 'REFPROP_9'
    
    if symlink_target.exists():
        if symlink_target.is_symlink():
            print(f"✓ Symlink already exists: {symlink_target}")
            return True
        else:
            print(f"⚠ {symlink_target} exists but is not a symlink")
            return False
    
    try:
        os.symlink(refprop9_path, symlink_target)
        print(f"✓ Created symlink: {symlink_target} -> {refprop9_path}")
        return True
    except OSError as e:
        print(f"✗ Failed to create symlink: {e}")
        return False


def verify_refprop9():
    """Verify REFPROP_9 can be imported."""
    import sys
    
    # Add common paths
    paths_to_check = [
        os.path.join(os.path.expanduser('~'), 'HyFlux_Hx', 'Hyflux', 'REFPROP_9'),
        os.path.join(Path(__file__).parent.parent, 'REFPROP_9'),
    ]
    
    for path in paths_to_check:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    try:
        import REFPROP_9
        print(f"✓ REFPROP_9 imported successfully")
        print(f"  Module: {REFPROP_9}")
        print(f"  Location: {REFPROP_9.__file__ if hasattr(REFPROP_9, '__file__') else 'N/A'}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import REFPROP_9: {e}")
        return False


def main():
    """Main setup function."""
    print("="*60)
    print("REFPROP_9 Setup from HyFlux_Hx Repository")
    print("="*60)
    print()
    
    # Option 1: Clone repository
    print("Step 1: Clone HyFlux_Hx repository...")
    repo_dir = clone_hyflux_repo()
    
    if repo_dir:
        print()
        print("Step 2: Create symlink in project...")
        create_symlink(repo_dir, None)
    
    print()
    print("Step 3: Verify REFPROP_9 import...")
    verify_refprop9()
    
    print()
    print("="*60)
    print("Setup complete!")
    print("="*60)
    print()
    print("You can now use REFPROP_9 in your configuration:")
    print("  fluid = FluidConfig(")
    print("      use_real_properties=True,")
    print("      property_backend='REFPROP',")
    print("      ...")
    print("  )")


if __name__ == "__main__":
    main()

