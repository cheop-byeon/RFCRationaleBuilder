import os
import sys
import git
import csv
import pathlib
import logging

csv.field_size_limit(sys.maxsize)

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Constants
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
VALID_TYPES = {'xml', 'md', 'mkd'}
EXCLUDED_NAMES = {'README', 'CONTRIBUTING', 'testing methodology'}

def get_valid_file_extensions(repo, commit_id):
    """Get file extensions for valid file types (xml, md, mkd) in a commit."""
    try:
        commit = repo.commit(commit_id)
        return [pathlib.Path(path).suffix[1:] for path in commit.stats.files.keys()]
    except Exception as e:
        logger.warning(f"Error checking file types for {commit_id}: {e}")
        return []


def has_valid_file_type(repo, commit_id):
    """Check if commit contains any valid file types."""
    extensions = get_valid_file_extensions(repo, commit_id)
    return any(ext in VALID_TYPES for ext in extensions)


def get_commit_indices(repo, commit_ids):
    """Get start and end indices of commits with valid file types."""
    if not commit_ids:
        return -1, -1
    
    start_idx = 0
    end_idx = len(commit_ids) - 1
    
    # Find first commit with valid file type
    while start_idx <= end_idx and not has_valid_file_type(repo, commit_ids[start_idx]):
        start_idx += 1
    
    # Find last commit with valid file type
    while end_idx >= start_idx and not has_valid_file_type(repo, commit_ids[end_idx]):
        end_idx -= 1
    
    if start_idx <= end_idx:
        return start_idx, end_idx
    return -1, -1


def restore_file_snapshots(repo_path, commit_ids, simplify=False):
    """Restore file snapshots for each commit with valid file types."""
    try:
        repo = git.Repo(repo_path)
        start_idx, end_idx = get_commit_indices(repo, commit_ids)
        
        if start_idx == -1 or end_idx == -1:
            logger.info(f"No valid files found in commits for {repo_path}")
            return
        
        if simplify:
            save_commit_files(repo, start_idx, commit_ids, repo_path)
            save_commit_files(repo, end_idx, commit_ids, repo_path)
        else:
            for idx in range(start_idx, end_idx + 1):
                save_commit_files(repo, idx, commit_ids, repo_path)
    except Exception as e:
        logger.error(f"Error processing {repo_path}: {e}")
    
def save_commit_files(repo, idx, commit_ids, repo_path):
    """Save before/after file snapshots for a commit."""
    try:
        commit_id = commit_ids[idx]
        commit = repo.commit(commit_id)
        parent = commit.parents[0] if commit.parents else EMPTY_TREE_SHA
        
        output_dir = os.path.join(repo_path, "snapshots") # any name that explicitly indicate the directory is for storing file snapshots
        os.makedirs(output_dir, exist_ok=True)
        
        diff_index = commit.diff(parent)
        
        for diff_item in diff_index.iter_change_type('M'):
            file_name = pathlib.Path(diff_item.a_blob.path).stem
            file_type = pathlib.Path(diff_item.a_blob.path).suffix[1:]
            
            # Skip excluded file names and non-valid types
            if file_name in EXCLUDED_NAMES or file_type not in VALID_TYPES:
                continue
            
            # Save before version
            before_path = os.path.join(output_dir, f'a_{commit_id}_{file_name}.{file_type}')
            if not os.path.exists(before_path):
                try:
                    content = diff_item.a_blob.data_stream.read().decode('utf-8')
                    with open(before_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    logger.warning(f"Error saving before version for {file_name}: {e}")
            
            # Save after version
            after_path = os.path.join(output_dir, f'b_{commit_id}_{file_name}.{file_type}')
            if not os.path.exists(after_path):
                try:
                    content = diff_item.b_blob.data_stream.read().decode('utf-8')
                    with open(after_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                except Exception as e:
                    logger.warning(f"Error saving after version for {file_name}: {e}")
    except Exception as e:
        logger.error(f"Error processing commit {commit_ids[idx]}: {e}")

def main():
    """Main entry point for restoring file snapshots from commits."""
    if len(sys.argv) > 2:
        sys.exit(f"Usage: {sys.argv[0]} [repo_path]")
    
    if len(sys.argv) == 1:
        sys.exit(f"Usage: {sys.argv[0]} <repo_path> [csv_file]")
    
    repo_path = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) == 3 else f"{repo_path}.csv"
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    logger.info(f"Processing {repo_path} with {csv_file}")
    
    with open(csv_file, "r") as f:
        reader = csv.reader(f, delimiter=",")
        next(reader)  # Skip header
        
        for row_num, row in enumerate(reader, start=2):
            if len(row) < 5:
                logger.warning(f"Row {row_num} has insufficient columns")
                continue
            
            pr_commit_ids_str = row[3]
            merged_commit_ids_str = row[4]
            
            pr_commit_ids = [cid.strip() for cid in pr_commit_ids_str.split(',') if cid.strip()]
            merged_commit_ids = [cid.strip() for cid in merged_commit_ids_str.split(',') if cid.strip()]
            
            # Use merged commits if available, otherwise use PR commits
            commit_ids = merged_commit_ids if merged_commit_ids else pr_commit_ids
            
            if commit_ids:
                restore_file_snapshots(repo_path, commit_ids)
    
    logger.info(f"Completed processing {repo_path}")

if __name__ == "__main__":
    sys.exit(main())