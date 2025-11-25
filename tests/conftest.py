import os
import sys

# Add both the repository root and src directory to path
repo_root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/..")
src_dir = os.path.join(repo_root, "src")

sys.path.insert(0, repo_root)
sys.path.insert(0, src_dir)
