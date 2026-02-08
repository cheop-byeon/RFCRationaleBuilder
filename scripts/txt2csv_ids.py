import os
import sys
import git
import pygit2
import csv
import textwrap
import re
import argparse
from textnorm import normalize_space
import pandas as pd
import hashlib
import logging
import glob
# from io import StringIO
from markdown import Markdown
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Precompile markdown converter for efficiency
_MARKDOWN_CONVERTER = Markdown(output_format="html")
_MARKDOWN_CONVERTER.stripTopLevelTags = False

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

def txt2csv(repo_path, draft_names, commitIds, pr, comments):
    """Extract paragraphs from git diffs and write to CSV"""
    if not commitIds or commitIds == ['']:
        logger.warning(f"No valid commits for PR {pr}")
        return
    
    repo_pygit2 = pygit2.init_repository(repo_path)
    repo = git.Repo(repo_path)
    start_index, end_index = get_commit_indices(repo, commitIds)
    
    if start_index == -1 or end_index == -1:
        logger.warning(f"No valid file types found in commits for PR {pr}")
        return
    
    commit_start = commitIds[start_index]
    commit_end = commitIds[end_index]
    out_folder = "simplified_xmls"
    
    logger.info(f"Processing PR {pr} with commits {commit_start} to {commit_end}")
    logger.info(f"Draft names: {draft_names}")
    
    for draft_name in draft_names:
        start_file_pattern = f"{repo_path}/{out_folder}/b_{commit_start}_{draft_name}.*.out.raw.txt"
        end_file_pattern = f"{repo_path}/{out_folder}/a_{commit_end}_{draft_name}.*.out.raw.txt"
        
        start_files = glob.glob(start_file_pattern)
        end_files = glob.glob(end_file_pattern)
        
        if not start_files or not end_files:
            logger.warning(f"Missing files for draft {draft_name}: found {len(start_files)} afiles, {len(end_files)} bfiles")
            continue
        
        # Use first matching file for each
        start_file = start_files[0]
        end_file = end_files[0]
        
        try:
            start_id = repo_pygit2.create_blob_fromdisk(start_file)
            end_id = repo_pygit2.create_blob_fromdisk(end_file)
            output = repo_pygit2.diff(start_id, end_id, cached=False, flags=0, context_lines=10, interhunk_lines=10) # for "whole" paragraph
            
            logger.info(f"Processing draft {draft_name}: found {len(output.hunks)} hunks")
            
            for hunk in output.hunks:
                block_s = get_block(start_file, hunk.old_start, hunk.old_lines)
                paras_s, filtered_paras_s = get_paragraphs(block_s)
                
                block_e = get_block(end_file, hunk.new_start, hunk.new_lines)
                paras_e, filtered_paras_e = get_paragraphs(block_e)
                
                logger.debug(f"Hunk extracted {len(paras_s)} paras_s and {len(paras_s)} paras_e")
                
                # Calculate block hash from all paragraphs
                all_paras_hash = [hashlib.sha256(p.encode('utf-8')).hexdigest() for p in paras_s + paras_e]
                block_hash = all_paras_hash
                
                if paras_s or paras_e:
                    write2csv(repo_path, paras_s, paras_e, pr, comments, commit_start, commit_end, draft_name, block_hash, filtered=False)
                if filtered_paras_s or filtered_paras_e:
                    write2csv(repo_path, filtered_paras_s, filtered_paras_e, pr, comments, commit_start, commit_end, draft_name, block_hash, filtered=True)
        except Exception as e:
            logger.error(f"Error processing draft {draft_name} for PR {pr}: {e}") 

def get_paragraphs(block):
    """Extract paragraphs from a block of text using simple heuristics."""
    exp_header = r"^(\d{1,2}\.(\d{1}\.)*) ?([^\W\d]+)?"
    
    paras_raw = list(filter(lambda x: len(x.strip()) != 0, block.split('\n\n')))
    paras = []
    filtered_paras = []
    
    for para_raw in paras_raw:
        lines = [line.strip() for line in para_raw.splitlines() if len(line.strip()) != 0 and len(re.findall(exp_header, line)) == 0]
        if len(lines) == 0:
            continue
        
        sentences = "\n".join(lines)
        sentences = re.sub(r"\[\[(.*?)\]\]", '', sentences)  # Remove double bracket patterns
        
        first_line = lines[0]
        first_letter = first_line[0]
        last_line = lines[-1]
        last_symbol = last_line[-1] if len(last_line) >= 1 else None
        
        # Simple rule: valid paragraphs start with letter/digit/quote and end with punctuation
        if (first_letter.isalpha() or first_letter.isdigit() or first_letter in '"\'') and last_symbol in '.?,':
            paras.append(sentences)
        else:
            filtered_paras.append(sentences)
    
    return paras, filtered_paras

def write2csv(repo_path, para_s, para_e, pr, comments, commit_s, commit_e, draft_name, hash=None, filtered=False):
    """Write paragraph data to CSV file"""
    sample_file = f"{repo_path}.abcd{''.join('.filter' if filtered else '')}.csv"
    sample_headers = ['pr', 'para', 'commit_s', 'commit_e', 'draft', 'para_hash', 'block_hash', 'AB']
    file_exists = os.path.isfile(sample_file)
    
    # Process comments once, the processed info can be saved into the following csv, but we remove it for now to save csv file size.
    comments_text, _, urls = process_comments(comments)
    print("write a file!")
    with open(sample_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, delimiter='\t', lineterminator='\n', fieldnames=sample_headers)
        if not file_exists:
            writer.writeheader()
        
        # Write para_a and para_b in single loop
        for para, ab_flag in [(p, 'A') for p in para_s if p.strip()] + [(p, 'B') for p in para_e if p.strip()]:
            writer.writerow({
                'pr': pr,
                'para': para,
                'commit_s': commit_s,
                'commit_e': commit_e,
                'draft': draft_name,
                'para_hash': hashlib.sha256(para.encode('utf-8')).hexdigest(),
                'block_hash': hash,
                'AB': ab_flag
            })
        print("write a file!")


def remove_emoji(text):
    """Remove emoji from text"""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def strip_markdown(text):
    """Remove markdown formatting using precompiled converter"""
    html = _MARKDOWN_CONVERTER.convert(text)
    # Strip HTML tags to get plain text
    return re.sub(r'<[^>]+>', '', html)

def remove_headlines(text):
    """Remove RFC section heading patterns"""
    text = re.sub(r'^\s*\d*\.\s*§\s*\d+(\.\d+)*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*§\s*\d+(\.\d+)*', '', text, flags=re.MULTILINE)
    return text

# Precompile regex patterns for efficiency
_BOT_COMMENT_PATTERNS = [
    r"Closes:?\s+#\d+.?",
    r"closes:?\s+#\d+.?",
    r"Fix:?\s+#\d+.?",
    r"fix:?\s+#\d+.?",
    r"Resolves:?\s+#\d+.?",
    r"resolves:?\s+#\d+.?",
]

def process_comments(txt):
    """Process and clean comment text"""
    if pd.isna(txt) or not isinstance(txt, str):
        return "", 0, ""
    
    result = []
    extracted_urls = []
    paras = txt.split("-------------------------------------------------------------------------------")
    
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    website_regex = r'http[s]?://\S+'
    email_address_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    html_pattern = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
    
    for para in paras:
        # Remove quotes and HTML
        para = re.sub(r'^\s*>\s+(.+?)\s*$', '', para, flags=re.MULTILINE)
        para = re.sub(r"(?:>+|\b>)\s*", '', para, flags=re.MULTILINE)
        para = re.sub(html_pattern, '', para, flags=re.MULTILINE)
        
        # Remove markdown code blocks
        para = re.sub(r'```.*?```', '', para, flags=re.DOTALL)
        para = re.sub(r"'''.*?'''", '', para, flags=re.DOTALL)
        para = re.sub(r'~~~.*?~~~', '', para, flags=re.DOTALL)
        
        # Remove headings
        para = re.sub(r'^#+\s+(.+?)\s*$', '', para, flags=re.MULTILINE)
        para = re.sub(r'^\s*=+\s*$', '', para, flags=re.MULTILINE)
        para = re.sub(r'^\s*-+\s*$', '', para, flags=re.MULTILINE)
        
        # Remove RFC headers and section headings
        para = remove_headlines(para)
        para = re.sub(r"^(\d{1,2}\.(\d{1}\.)*) ?([^\W\d]+)?", '', para, flags=re.MULTILINE)
        
        # Remove strike through and formatting
        para = re.sub(r"\~\~(.+?)\~\~", '', para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'\*\*(.+?)\*\*', r'\1', para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'__(.+?)__', r'\1', para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'\*(.+?)\*', r'\1', para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'_(.+?)_', r'\1', para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'\`(.+?)\`', '', para, flags=re.MULTILINE)
        
        # Remove bot comment patterns
        for pattern in _BOT_COMMENT_PATTERNS:
            para = re.sub(pattern, '', para, flags=re.MULTILINE | re.DOTALL)
        
        # Remove list markers
        para = re.sub(r'^\s*\d+\.\s+(.+?)\s*$', r'\1', para, flags=re.MULTILINE)
        para = re.sub(r'^\s*[\*\+-]\s+(.+?)\s*$', r'\1', para, flags=re.MULTILINE)
        
        # Remove horizontal lines and links
        para = re.sub(r'^\s*[-*_]{3,}\s*$', '', para, flags=re.MULTILINE)
        para = re.sub(r'\[(.+?)\]\((.+?)\)', '', para, flags=re.MULTILINE)
        para = re.sub(r'!\[(.+?)\]\((.+?)\)', '', para, flags=re.MULTILINE)
        
        # Extract and remove URLs
        extracted_urls += re.findall(url_regex, para)
        para = re.sub(website_regex, '', para)
        
        # Remove email and contact info
        para = re.sub(email_address_regex, '', para)
        para = re.sub(r"On \w{3}, \w{3} \d{1,2}, \d{4} at \d{1,2}:\d{2} (AM|PM)", '', para, flags=re.MULTILINE)
        para = re.sub(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b', '', para)
        
        # Remove formulas and tables
        para = re.sub(r"\${1,2}.+?\${1,2}", "", para)
        para = re.sub(r"\|.*\|", "", para)
        
        # Remove mentions and references
        para = re.sub(r"@\S+", '', para)
        para = re.sub(r'#\d+', '', para)
        para = re.sub(r"(Issue|PR):?\s*#\d+", '', para, flags=re.MULTILINE)
        
        # Remove emojis and hashes
        para = remove_emoji(para)
        para = re.sub(r'^[a-fA-F0-9]{64}$', "", para)
        para = re.sub(r'^[a-fA-F0-9]{32}$', "", para)
        para = re.sub(r'\b[0-9a-f]{5,40}\b', '', para)
        
        # Clean whitespace
        para = re.sub(r'\s+', ' ', para)
        output = "\n".join([line.strip() for line in para.splitlines()])
        output = normalize_space(output)
        result.append(output)
    
    commentary = "\n".join(result).strip()
    commentary = strip_markdown(commentary)
    
    return commentary, len(commentary.split()), "\n".join(extracted_urls)

def get_block(file, start_line, lines):
    """Extract a block from a file with context"""
    with open(file) as f:
        content = f.readlines()
    
    length = len(content)
    default_context_lines = 3
    context_lines = 10
    start_line = max(0, start_line - 1 + default_context_lines)
    end_line = min(length - 1, start_line + (lines - 2 * default_context_lines) - 1)
    
    # Reduce context lines if needed
    while (end_line + context_lines > length - 1 or start_line - context_lines < 0) and context_lines > 0:
        context_lines -= 1
    
    # Find paragraph boundaries
    a_paragraph_start_line = start_line
    a_paragraph_end_line = end_line
    
    for line_idx in range(start_line - context_lines, start_line):
        if line_idx >= 0 and content[line_idx] == '\n':
            a_paragraph_start_line = line_idx + 1
    
    for line_idx in range(end_line + context_lines, end_line, -1):
        if line_idx < length and content[line_idx] == '\n':
            a_paragraph_end_line = line_idx - 1
    
    return textwrap.dedent("".join(content[a_paragraph_start_line:a_paragraph_end_line+1]))

def remove_duplicate(file, save_duplicates=True):
    """Remove duplicate paragraphs from CSV file.
    
    Args:
        file: Path to the CSV file
        save_duplicates: If True, save duplicate rows to .duplicate.csv file
    """
    df = pd.read_csv(file, sep='\t')
    df['paragraph'] = df['para'].str.strip().str.lower()
    
    duplicate_count = df.duplicated(subset=['paragraph', 'pr'], keep=False).sum()
    logger.info(f"{file}: {duplicate_count} duplicates found")
    
    # Write duplicates to a separate file if requested
    if save_duplicates:
        filename = file.rsplit('.', 2 if file.endswith('.filter.csv') else 1)[0]
        duplicated_rows = df[df.duplicated(subset=['paragraph', 'pr'], keep='last')]
        if len(duplicated_rows) > 0:
            duplicated_rows.to_csv(f'{filename}.duplicate.csv', sep='\t', index=False, mode='a', header=False)
            logger.info(f"Saved {len(duplicated_rows)} duplicate rows to {filename}.duplicate.csv")
    
    # Keep only unique rows
    df.drop_duplicates(subset=['paragraph', 'pr'], inplace=True, keep=False)
    
    if len(df) != 0:
        df.to_csv(file, sep='\t', index=False)
    else:
        os.remove(file)
    
    return duplicate_count

def main():
    """Main entry point for txt2csv processing"""
    parser = argparse.ArgumentParser(description='Extract code rationale from GitHub issues/PRs')
    parser.add_argument('repo_path', nargs='?', default='drafts', help='Path to repository (default: drafts)')
    parser.add_argument('--save-filtered', action='store_true', default=True, help='Save .filter.csv file (default: True)')
    parser.add_argument('--no-save-filtered', action='store_false', dest='save_filtered', help='Do not save .filter.csv file')
    parser.add_argument('--save-duplicates', action='store_true', default=True, help='Save .duplicate.csv file (default: True)')
    parser.add_argument('--no-save-duplicates', action='store_false', dest='save_duplicates', help='Do not save .duplicate.csv file')
    
    args = parser.parse_args()
    
    try:
        repo_path = os.path.abspath(args.repo_path) if os.path.isabs(args.repo_path) else os.path.join(os.getcwd(), args.repo_path)
        local_path = f"{repo_path}.csv"
        
        logger.info(f"Looking for CSV file: {local_path}")
        logger.info(f"Save filtered: {args.save_filtered}, Save duplicates: {args.save_duplicates}")
        
        if not os.path.isfile(local_path):
            logger.error(f"CSV file not found: {local_path}")
            return 1
        
        out_dir = os.path.join(repo_path, 'simplified_xmls')
        logger.info(f"Looking for output directory: {out_dir}")
        
        if not os.path.isdir(out_dir):
            logger.error(f"Output directory not found: {out_dir}")
            return 1
        
        logger.info(f"Starting processing from {local_path}")
        
        with open(local_path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header
            for line in reader:
                pr = line[0]
                comments = line[7]
                merged_commitIds = [cid.strip() for cid in line[4].split(',') if cid.strip()]
                pr_commitIds = [cid.strip() for cid in line[3].split(',') if cid.strip()]
                
                logger.info(f"Processing PR {pr}")
                
                # Get draft names from output directory
                draft_names = set()
                for name in os.listdir(out_dir):
                    try:
                        file = name.split('_')[2]
                        draft = file.split('.')[0]
                        draft_names.add(draft)
                    except IndexError:
                        continue
                
                if not draft_names:
                    logger.warning(f"No drafts found for PR {pr}")
                    continue
                
                # Process with merged commits if available, else use PR commits
                commit_ids = merged_commitIds if merged_commitIds != [''] else pr_commitIds
                txt2csv(repo_path, draft_names, commit_ids, pr, comments)
        
        # Remove duplicates from output files based on options
        files_to_process = ['.abcd.csv']
        if args.save_filtered:
            files_to_process.append('.abcd.filter.csv')
        
        for suffix in files_to_process:
            output_file = f"{repo_path}{suffix}"
            if os.path.isfile(output_file):
                remove_duplicate(output_file, save_duplicates=args.save_duplicates)
        
        logger.info("Processing complete")
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())