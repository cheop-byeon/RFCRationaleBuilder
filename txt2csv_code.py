import os
import sys
import git
import pygit2
import csv
import re
from textnorm import normalize_space
import pandas as pd
import hashlib
import multiprocessing
import argparse
import logging
from markdown import Markdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module Constants
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
VALID_TYPES = {'xml', 'md', 'mkd'}

# Precompiled Markdown converter for efficiency
_MARKDOWN_CONVERTER = Markdown(output_format="html")
_MARKDOWN_CONVERTER.stripTopLevelTags = False

# Precompiled bot comment patterns for efficiency
_BOT_COMMENT_PATTERNS = [
    r"Closes:?\s+#\d+.?",
    r"closes:?\s+#\d+.?",
    r"Fix:?\s+#\d+.?",
    r"fix:?\s+#\d+.?",
    r"Resolves:?\s+#\d+.?",
    r"resolves:?\s+#\d+.?",
    r"Issue:?\s+\S+/\S+#\d+.?",
    r"Closed via\s+#\d+.?",
    r"Closed by\s+#\d+.?",
]

# Precompiled regex patterns
_EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
_WEBSITE_REGEX = re.compile(r'https?://(?:www\.|(?!www))[^\s.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,}')
_SHA256_REGEX = re.compile(r'^[a-fA-F0-9]{64}$')
_MD5_REGEX = re.compile(r'^[a-fA-F0-9]{32}$')

def issue2pr(kuber, repo_path):
    repo_pygit2 = pygit2.init_repository(repo_path) 
    repo = git.Repo(repo_path) 

    kuber_issues = kuber[kuber['type'] == 'issue']
    kuber_prs = kuber[(kuber['type'] == 'pr') & (kuber['state'] == 'merged')]
    kuber_issues_we_want_most = kuber_issues[kuber_issues['closedByPullRequestsReferences_numbers'].notnull()]
    kuber_issues_we_want_next = kuber_issues[(kuber_issues['associatedPullRequests_numbers'].notnull()) & (kuber_issues['closedByPullRequestsReferences_numbers'].isnull())]
    kuber_issues_we_want = pd.concat([kuber_issues_we_want_most, kuber_issues_we_want_next], axis=0)

    sample_file ='{}.merged.csv'.format(repo_path)
    sample_headers = ['issue', 'issue_state', 'pr', 'pr_state', 'para', 'commit_s','commit_e', 'comments', 'block_hash', 'url']
    file_exists = os.path.isfile(os.path.join(repo_path, sample_file))

    with open(os.path.join(repo_path, sample_file), 'a', newline='') as sample_file:
        writer = csv.DictWriter(sample_file, delimiter='\t', lineterminator='\n',fieldnames=sample_headers)
        if not file_exists:
            writer.writeheader()

        for _, issue_row in kuber_issues_we_want.iterrows():
            issue_number, id, type, issue_state, pr_commits, merged_commits, parents, sourceIssuesReferences_numbers, closingIssuesReferences_numbers, trackedIssues_numbers, \
            trackedInIssues_numbers, closedByPullRequestsReferences_numbers, associatedPullRequests_numbers, sourceIssuesReferences_ids, closingIssuesReferences_ids, \
            trackedIssues_ids, trackedInIssues_ids, closedByPullRequestsReferences_ids, associatedPullRequests_ids, labels, comments = issue_row

            if pd.isna(associatedPullRequests_numbers):
                vs = [int(should_be_pr_number.replace('#', '')) for should_be_pr_number in closedByPullRequestsReferences_numbers.split(',')]
            elif pd.isna(closedByPullRequestsReferences_numbers):
                vs = [int(should_be_pr_number.replace('#', '')) for should_be_pr_number in associatedPullRequests_numbers.split(',')]
            else:
                continue

            pr_numbers = [v for v in vs if v in kuber_prs['number'].values]

            for pr_number in pr_numbers:
                pr_row = kuber_prs[kuber_prs['number'] == pr_number]
                merged_commit, not_merged_commits = pr_row['merged_commits'].item(), pr_row['pr_commits'].item()
                
                if pd.notnull(merged_commit):
                    try:
                        repo.commit(merged_commit)
                    except:
                        continue

                    commit = repo_pygit2.revparse_single(merged_commit)
                    parents_commit = commit.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(comments)
                            writer.writerow({'issue': issue_number, 'issue_state': issue_state, 'pr':pr_number, 'pr_state': pr_row['state'].values[0], 'para':block_diff, \
                                                    'commit_s': merged_commit, 'commit_e': merged_commit, 'comments': comments_text, 'block_hash': block_hash, 'url': urls})

                elif pd.notnull(not_merged_commits):
                    commits = not_merged_commits.split(', ')
                    valid_commit_hashes_indexes = []
                    for idx, c in enumerate(commits):
                        try:
                            repo.commit(c)
                            valid_commit_hashes_indexes.append(idx)
                        except:
                            continue

                    if len(valid_commit_hashes_indexes) == 0:
                        continue

                    commit1, commit2 = commits[valid_commit_hashes_indexes[0]], commits[valid_commit_hashes_indexes[-1]]
                    commit_e = repo_pygit2.revparse_single(commit2)
                    commit_s = repo_pygit2.revparse_single(commit1)
                    parents_commit = commit_s.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit_e, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(comments)
                            writer.writerow({'issue': issue_number, 'issue_state': issue_state, 'pr':pr_number, 'pr_state': pr_row['state'].values[0], 'para':block_diff, \
                                                    'commit_s': commit1, 'commit_e': commit2, 'comments': comments_text,'block_hash': block_hash, 'url': urls})
                            
def issue2pr2(kuber_issues_we_want, kuber_prs, repo_path, workid):
    logging.info(f"Worker {workid} is processing {len(kuber_issues_we_want)} rows")
    repo_pygit2 = pygit2.init_repository(repo_path) 
    repo = git.Repo(repo_path) 

    sample_file ='{}.{}.merged.csv'.format(repo_path, workid)
    sample_headers = ['issue', 'issue_state', 'pr', 'pr_state', 'para', 'commit_s','commit_e', 'comments', 'block_hash', 'url']
    file_exists = os.path.isfile(os.path.join(repo_path, sample_file))

    with open(os.path.join(repo_path, sample_file), 'a', newline='') as sample_file:
        writer = csv.DictWriter(sample_file, delimiter='\t', lineterminator='\n',fieldnames=sample_headers)
        if not file_exists:
            writer.writeheader()

        for _, issue_row in kuber_issues_we_want.iterrows():
            issue_number, id, type, issue_state, pr_commits, merged_commits, parents, sourceIssuesReferences_numbers, closingIssuesReferences_numbers, trackedIssues_numbers, \
            trackedInIssues_numbers, closedByPullRequestsReferences_numbers, associatedPullRequests_numbers, sourceIssuesReferences_ids, closingIssuesReferences_ids, \
            trackedIssues_ids, trackedInIssues_ids, closedByPullRequestsReferences_ids, associatedPullRequests_ids, labels, comments = issue_row

            if pd.isna(associatedPullRequests_numbers):
                vs = [int(should_be_pr_number.replace('#', '')) for should_be_pr_number in closedByPullRequestsReferences_numbers.split(',')]
            elif pd.isna(closedByPullRequestsReferences_numbers):
                vs = [int(should_be_pr_number.replace('#', '')) for should_be_pr_number in associatedPullRequests_numbers.split(',')]
            else:
                continue
            
            pr_numbers = [v for v in vs if v in kuber_prs['number'].values]

            for pr_number in pr_numbers:
                pr_row = kuber_prs[kuber_prs['number'] == pr_number]
                merged_commit, not_merged_commits = pr_row['merged_commits'].item(), pr_row['pr_commits'].item()
                
                if pd.notnull(merged_commit):
                    try:
                        repo.commit(merged_commit)
                    except:
                        continue

                    commit = repo_pygit2.revparse_single(merged_commit)
                    parents_commit = commit.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(comments)
                            writer.writerow({'issue': issue_number, 'issue_state': issue_state, 'pr':pr_number, 'pr_state': pr_row['state'].values[0], 'para':block_diff, \
                                                    'commit_s': merged_commit, 'commit_e': merged_commit, 'comments': comments_text, 'block_hash': block_hash, 'url': urls})

                elif pd.notnull(not_merged_commits):
                    commits = not_merged_commits.split(', ')
                    valid_commit_hashes_indexes = []
                    for idx, c in enumerate(commits):
                        try:
                            repo.commit(c)
                            valid_commit_hashes_indexes.append(idx)
                        except:
                            continue

                    if len(valid_commit_hashes_indexes) == 0:
                        continue

                    commit1, commit2 = commits[valid_commit_hashes_indexes[0]], commits[valid_commit_hashes_indexes[-1]]
                    commit_e = repo_pygit2.revparse_single(commit2)
                    commit_s = repo_pygit2.revparse_single(commit1)
                    parents_commit = commit_s.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit_e, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(comments)
                            writer.writerow({'issue': issue_number, 'issue_state': issue_state, 'pr':pr_number, 'pr_state': pr_row['state'].values[0], 'para':block_diff, \
                                                    'commit_s': commit1, 'commit_e': commit2, 'comments': comments_text,'block_hash': block_hash, 'url': urls})
                            
def pr2issue(kuber, repo_path, status='merged'):
    repo_pygit2 = pygit2.init_repository(repo_path) 
    repo = git.Repo(repo_path) 

    kuber_issues = kuber[kuber['type'] == 'issue']
    kuber_prs = kuber[(kuber['type'] == 'pr') & (kuber['state'] == status)]
    kuber_prs_we_want = kuber_prs[kuber_prs['closingIssuesReferences_numbers'].notnull()]
    print(len(kuber_prs_we_want))

    sample_file ='{}.merged.csv'.format(repo_path)
    sample_headers = ['issue', 'issue_state', 'pr', 'pr_state', 'para', 'commit_s','commit_e', 'comments', 'block_hash', 'url']
    file_exists = os.path.isfile(os.path.join(repo_path, sample_file))

    with open(os.path.join(repo_path, sample_file), 'a', newline='') as sample_file:
        writer = csv.DictWriter(sample_file, delimiter='\t', lineterminator='\n',fieldnames=sample_headers)
        if not file_exists:
            writer.writeheader()
           
        for _, pr_row in kuber_prs_we_want.iterrows():
            number, id, type, state, pr_commits, merged_commits, parents, sourceIssuesReferences_numbers, closingIssuesReferences_numbers, trackedIssues_numbers, \
            trackedInIssues_numbers, closedByPullRequestsReferences_numbers, associatedPullRequests_numbers, sourceIssuesReferences_ids, closingIssuesReferences_ids, \
            trackedIssues_ids, trackedInIssues_ids, closedByPullRequestsReferences_ids, associatedPullRequests_ids, labels, comments = pr_row

            vs = [int(should_be_issue_number.replace('#', '')) for should_be_issue_number in closingIssuesReferences_numbers.split(',')]
            issue_numbers = [v for v in vs if v in kuber_issues['number'].values]

            if len(issue_numbers) == 0:
                continue

            for issue_number in issue_numbers:
                issue_row = kuber_issues[kuber_issues['number'] == issue_number]
                merged_commit, not_merged_commits = merged_commits, pr_commits
                
                if pd.notnull(merged_commit):
                    commit = repo_pygit2.revparse_single(merged_commit)
                    parents_commit = commit.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(issue_row['comments'])
                            writer.writerow({'issue': issue_row['number'], 'issue_state': issue_row['state'].values[0], 'pr':number, 'pr_state': state, 'para':block_diff, \
                                                    'commit_s': merged_commit, 'commit_e': merged_commit, 'comments': comments_text,'block_hash': block_hash, 'url': urls})

                elif pd.notnull(not_merged_commits):
                    commits = not_merged_commits.split(', ')
                    valid_commit_hashes_indexes = []
                    for idx, c in enumerate(commits):
                        try:
                            repo.commit(c)
                            valid_commit_hashes_indexes.append(idx)
                        except:
                            continue

                    if len(valid_commit_hashes_indexes) == 0:
                        continue
                    
                    commit1, commit2 = commits[valid_commit_hashes_indexes[0]], commits[valid_commit_hashes_indexes[-1]]
                    commit_e = repo_pygit2.revparse_single(commit2)
                    commit_s = repo_pygit2.revparse_single(commit1)
                    parents_commit = commit_s.parents[0]
                    diff = repo_pygit2.diff(parents_commit, commit_e, cached=False, flags = 0, context_lines = 3, interhunk_lines = 0)
                    for patch in diff:
                        for hunk in patch.hunks:
                            block_diff = get_block(hunk, ignore_deletions = False, ignore_context = False)
                            block_hash = hashlib.sha256(block_diff.encode('utf-8')).hexdigest()
                            comments_text, urls = process_comments(issue_row['comments'].values[0])
                            writer.writerow({'issue': issue_row['number'].values[0], 'issue_state': issue_row['state'].values[0], 'pr':number, 'pr_state': state, \
                                                    'para':block_diff, 'commit_s': commit1, 'commit_e': commit2, 'comments': comments_text,'block_hash': block_hash, 'url': urls})

def remove_headlines(text):
    """Remove RFC section heading patterns."""
    text = re.sub(r'^\s*\d*\.\s*§\s*\d+(\.\d+)*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*§\s*\d+(\.\d+)*', '', text, flags=re.MULTILINE)
    return text

def strip_markdown(text):
    """Remove markdown formatting using precompiled converter."""
    html = _MARKDOWN_CONVERTER.convert(text)
    return re.sub(r'<[^>]+>', '', html)

# Precompiled emoji pattern for efficiency
_EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"  # Dingbats
    u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def remove_emoji(text):
    """Remove emoji from text using precompiled pattern."""
    return _EMOJI_PATTERN.sub('', text)

def process_comments(txt):
    """Process and clean comment text."""
    if pd.isna(txt) or not isinstance(txt, str):
        return "", ""
    
    result = []
    extracted_urls = []
    paras = txt.split("-------------------------------------------------------------------------------")
    
    # Precompile regexes that are used repeatedly
    url_regex = re.compile(r'(https?://(?:www\.|(?!www))[^\s.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})')
    phone_regex = re.compile(r'^(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}$')
    html_pattern = re.compile(r"<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>")
    
    for para in paras:
        # Remove markdown code blocks
        para = re.sub(r'```.*?```|\'\'\'.*?\'\'\'|~~~.*?~~~', '', para, flags=re.DOTALL)
        
        # Remove headings
        para = re.sub(r'^#+\s+(.+?)\s*$|^\s*[=\-]+\s*$', '', para, flags=re.MULTILINE)
        
        # Remove section headings and numerics
        para = remove_headlines(para)
        para = re.sub(r"^(\d{1,2}\.(\d{1}\.)*) ?([^\W\d]+)?", '', para, flags=re.MULTILINE)
        
        # Remove formatting symbols
        para = re.sub(r'~~(.+?)~~|\*\*(.+?)\*\*|__(.+?)__|\.(.+?)\.|_(.+?)_', lambda m: next(g for g in m.groups() if g is not None), para, flags=re.MULTILINE | re.DOTALL)
        para = re.sub(r'`(.+?)`', '', para, flags=re.MULTILINE)
        
        # Remove block quotes and HTML
        para = re.sub(r'^\s*>\s+(.+?)\s*$|(?:>+|\b>)\s*', '', para, flags=re.MULTILINE)
        para = html_pattern.sub('', para)
        
        # Remove bot patterns
        for pattern in _BOT_COMMENT_PATTERNS:
            para = re.sub(pattern, '', para, flags=re.MULTILINE | re.DOTALL)
        
        # Remove list markers
        para = re.sub(r'^\s*\d+\.\s+(.+?)\s*$|^\s*[\*\+-]\s+(.+?)\s*$', lambda m: next((g for g in m.groups() if g is not None), ''), para, flags=re.MULTILINE)
        
        # Remove horizontal lines and links
        para = re.sub(r'^\s*[-*_]{3,}\s*$|\[(.+?)\]\((.+?)\)|!\[(.+?)\]\((.+?)\)', '', para, flags=re.MULTILINE)
        
        # Extract and remove URLs
        extracted_urls.extend(re.findall(url_regex, para))
        para = url_regex.sub('', para)
        
        # Remove email and phone
        para = _EMAIL_REGEX.sub('', para)
        para = re.sub(r'On \w{3}, \w{3} \d{1,2}, \d{4} at \d{1,2}:\d{2} (AM|PM)', '', para, flags=re.MULTILINE)
        para = phone_regex.sub('', para, flags=re.MULTILINE)
        
        # Remove formulas, tables, and special patterns
        para = re.sub(r'\${1,2}.+?\${1,2}|\|.*\||@\S+|#\d+|(Issue|PR):?\s*#\d+', '', para, flags=re.MULTILINE)
        
        # Remove emojis and hashes
        para = remove_emoji(para)
        para = _SHA256_REGEX.sub('', para)
        para = _MD5_REGEX.sub('', para)
        para = re.sub(r'\b[0-9a-f]{5,40}\b', '', para)
        
        # Clean whitespace
        para = re.sub(r'\s+', ' ', para)
        output = "\n".join([line.strip() for line in para.splitlines()])
        output = normalize_space(output)
        result.append(output)
    
    commentary = "\n".join(result).strip()
    commentary = strip_markdown(commentary)
    return commentary, "\n".join(extracted_urls)

def get_block(hunk: pygit2.DiffHunk, ignore_deletions: bool = True, ignore_context: bool=False) -> str:
    last_state: str = " "
    result: list[str] = []
    for line in hunk.lines:
        if ignore_deletions and line.origin == "-":
            continue
        if ignore_context and line.origin == " ":
            continue
        content = line.content.strip()
        if not content:
            continue
        if line.origin != last_state:
            if last_state == "+":
                result.append("</ins>")
            elif last_state == "-":
                result.append("</del>")
            if line.origin == "+":
                result.append("<ins>")
            elif line.origin == "-":
                result.append("<del>")
        result.append(content)
        last_state = line.origin
    if last_state == "+":
        result.append("</ins>")
    elif last_state == "-":
        result.append("</del>")

    return " ".join(result)



def split_dataframe(df, num_chunks):
    chunk_size = len(df) // num_chunks
    chunks = [df[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    # Handle any remaining rows
    if len(df) % num_chunks != 0:
        chunks.append(df[num_chunks*chunk_size:])
    
    return chunks # num_chunks or num_chunks + 1

def prepare_args(issues, num_chunks, prs, repo_path):
    issue_chunks = split_dataframe(issues, num_chunks)
    return [(chunk, prs, repo_path, workid) for workid, chunk in enumerate(issue_chunks)]

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--parallel', action="store_true", help='If use parallel features')
    parser.add_argument('--num_chunks', type=int, required=True, help='Number of chunks to split the DataFrame')
    parser.add_argument('--repo_path', type=str, required=True, help='Path to the repo')
    args = parser.parse_args()
    cwd = os.getcwd()
    repo_path = os.path.join(cwd, args.repo_path)
    local_path = os.path.join(cwd, repo_path + ".csv")
    
    # for kubernetes
    kuber = pd.read_csv(local_path, low_memory=False)
    # issue2pr(df, repo_path)

    kuber_issues = kuber[kuber['type'] == 'issue']
    kuber_prs = kuber[(kuber['type'] == 'pr') & (kuber['state'] == 'merged')]
    kuber_issues_we_want_most = kuber_issues[kuber_issues['closedByPullRequestsReferences_numbers'].notnull()]
    kuber_issues_we_want_next = kuber_issues[(kuber_issues['associatedPullRequests_numbers'].notnull()) & (kuber_issues['closedByPullRequestsReferences_numbers'].isnull())]
    kuber_issues_we_want = pd.concat([kuber_issues_we_want_most, kuber_issues_we_want_next], axis=0)

    if args.parallel == True:
        num_chunks = args.num_chunks # 9
        args = prepare_args(kuber_issues_we_want, num_chunks - 1, kuber_prs, repo_path) # 8 or 9
        pool = multiprocessing.Pool(processes=num_chunks) # 9
        pool.starmap(issue2pr2, args)
        pool.close()
        pool.join()

        combined_df = pd.DataFrame()
        for workid in range(num_chunks):
            file = '{}.{}.merged.csv'.format(repo_path, workid)
            df = pd.read_csv(file, sep='\t')
            combined_df = pd.concat([combined_df, df], ignore_index=True, axis=0)  # Combine DataFrames along rows
        print("1", len(combined_df))
        combined_df.drop_duplicates(subset=['issue', 'pr', 'block_hash'], keep='last')
        combined_df.to_csv('{}.combined.merged.csv'.format(repo_path), sep='\t')
        print("2", len(combined_df))
    else:
        issue2pr2(kuber_issues_we_want, kuber_prs, repo_path, 0)

if __name__ == "__main__":
    sys.exit(main())
