from shared.config_loader import load_config
import sys
from pathlib import Path

# Add project root to sys.path so shared module can be found
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def main():
    config = load_config()
    print("Loaded Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
        
    expected_python = "/Users/kurodakotaro/Documents/Stock/stock_forecast/.venv/bin/python"
    expected_venv = "/Users/kurodakotaro/Documents/Stock/stock_forecast/.venv"
    
    if config.get("python_path") == expected_python and config.get("venv_path") == expected_venv:
        print("\nVerification SUCCESS: Configuration matches expected values.")
    else:
        print("\nVerification FAILED: Configuration does not match expected values.")

    from shared.config_loader import apply_config
    apply_config(config)
    
    expected_extra = "/Users/kurodakotaro/Documents/Stock/stock_forecast/.venv/lib/python3.11/site-packages"
    if expected_extra in sys.path:
        print("Verification SUCCESS: Extra path added to sys.path.")
    else:
        print("Verification FAILED: Extra path NOT added to sys.path.")


if __name__ == "__main__":
    main()
