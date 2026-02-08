import argparse
from datetime import timezone
import datetime
import logging
import os
import pprint
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import csv
import requests
from dateutil.parser import parse as dateutil_parse
from email_reply_parser import EmailReplyParser
import git

ENV_GITHUB_TOKEN = "GITHUB_ACCESS_TOKEN"
GITHUB_ACCESS_TOKEN_PATHS = [
    os.path.expanduser(os.path.join("~", ".config", ".github-token")),
]

logformat = "[%(asctime)s] [%(levelname)s] %(msg)s"
logging.basicConfig(level=logging.INFO, format=logformat)
logger = logging.getLogger(__name__)

@dataclass
class GithubComment:
    """
    Represents a comment on an issue or PR.
    """

    user_login: str
    user_url: str
    created_at: datetime.datetime
    created_via_email: bool
    url: str
    body: str


@dataclass
class GithubIssue:
    """
    Represents an Issue or PR. The old Github API used to treat these as a
    single type of object, so initially I'm keeping it that way as we port
    this code to use GraphQL. It might make sense to separate out later.
    """

    user_login: str
    user_url: str
    pull_request: bool
    state: str
    body: str
    comments: List[GithubComment]
    number: int
    label_names: List[str]
    closingIssuesReferences_numbers: List[int]
    closingIssuesReferences_ids: List[int]
    pr_commits_ids: List[str]
    merged_commits_ids: List[str]
    commits_dates: List[datetime.datetime]
    parents_ids: List[str]
    
    title: str
    created_at: datetime.datetime
    url: str
    id: str
    sourceIssuesReferences_numbers: List[int] 
    sourceIssuesReferences_ids: List[str] # if it is an issue referenced another issue or pr, they might belong different repo.
    associatedPullRequests_numbers: List[int] # the issue was closed or referenced by commit, and commit was merged by pr
    associatedPullRequests_ids: List[str] # if it is an issue referenced another issue or pr, they might belong different repo.

    trackedInIssues_numbers: List[int] # A list of issues that track this issue.
    trackedIssues_numbers: List[int] # A list of issues tracked inside the current issue.
    closedByPullRequestsReferences_numbers: List[int] # List of open pull requests referenced from this issue.

    trackedInIssues_ids: List[str] # A list of issues that track this issue.
    trackedIssues_ids: List[str] # A list of issues tracked inside the current issue.
    closedByPullRequestsReferences_ids: List[str] # List of open pull requests referenced from this issue.

@dataclass
class GithubCommit:
    id: str
    committed_date: datetime.datetime
    associatedPullRequests_ids: List[str]
    associatedPullRequests_numbers: List[str]
    parents_ids: List[str]

@dataclass
class GithubRepo:
    """
    Root object representing a repo.
    """

    full_name: str
    url: str
    issues: List[GithubIssue]

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Export GitHub repository issues and pull requests to CSV file. "
                    f"Requires {ENV_GITHUB_TOKEN} environment variable or token in ~/.config/.github-token",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repo",
        help='Github repo to export, in format "owner/repo_name".',
        type=str,
        action="store",
    )
    parser.add_argument(
        "output_file_name",
        help="Output CSV file name.",
        type=str,
        action="store",
    )
    parser.add_argument(
        "--no-prs",
        help="Don't include pull requests in the export.",
        action="store_false",
        dest="include_prs",
    )
    parser.add_argument(
        "--no-closed-prs",
        help="Don't include closed pull requests in the export.",
        action="store_false",
        dest="include_closed_prs",
    )
    parser.add_argument(
        "--no-issues",
        help="Don't include issues in the export.",
        action="store_false",
        dest="include_issues",
    )
    parser.add_argument(
        "--no-closed-issues",
        help="Don't include closed issues in the export.",
        action="store_false",
        dest="include_closed_issues",
    )

    return parser.parse_args()


@dataclass
class GithubAPI:
    """
    Handles GraphQL API queries.
    """

    token: str = ""
    per_page: int = 10 

    _ENDPOINT = "https://api.github.com/graphql"
    _REPO_QUERY = """
        query(
          $owner: String!
          $repo: String!
          $issuePerPage: Int!
          $issueNextPageCursor: String
          $pullRequestPerPage: Int!
          $pullRequestNextPageCursor: String
          $issueStates: [IssueState!]
          $pullRequestStates: [PullRequestState!]
        ) {
          rateLimit {
            limit
            cost
            remaining
            resetAt
          }
          repository(owner: $owner, name: $repo) {
            nameWithOwner
            url
            issues (
              first: $issuePerPage
              after: $issueNextPageCursor
              filterBy: { states: $issueStates }
              orderBy: { field: CREATED_AT, direction: DESC }
            ) {
              totalCount
              pageInfo {
                endCursor
                hasNextPage
              }
              nodes {
                id
                number
                url
                title
                body
                state
                createdAt
                author {
                  login
                  url
                }
                labels(first: $issuePerPage) {
                  nodes {
                    name
                    url
                  }
                }
                trackedInIssues(first: $issuePerPage) {
                    totalCount
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                    nodes {
                        id
                        number
                    }
                }
                trackedIssues(first: $issuePerPage) {
                    totalCount
                    pageInfo {
                        endCursor
                        hasNextPage
                    }
                    nodes {
                        id
                        number
                    }
                }
                closedByPullRequestsReferences(first: $issuePerPage) {
                  totalCount
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                    nodes {
                        id
                        number
                    }
                }
                comments(first: $issuePerPage) {
                  totalCount
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                  nodes {
                    body
                    createdAt
                    createdViaEmail
                    url
                    author {
                      login
                      url
                    }
                  }
                }
                timelineItems(first: 10) {
                    nodes {
                        ... on CrossReferencedEvent {
                            source {
                                ... on Issue{
                                    id
                                    number
                                }
                                ... on PullRequest{
                                    id
                                    number
                                }
                            }
                        }
                        ... on ReferencedEvent {
                            commit {
                                oid,
                                committedDate,
                                associatedPullRequests(first: 10) {
                                    nodes {
                                        id,
                                        number
                                    }
                                }
                            }
                        }
                        ... on ClosedEvent {
                            closer {
                                ... on Commit {
                                    oid,
                                    committedDate,
                                    associatedPullRequests(first: 10) {
                                        nodes {
                                            id
                                            number
                                        }
                                    }
                                }
                            }
                            
                        }
                    }
                }
              }
            }
            pullRequests(
              first: $pullRequestPerPage
              after: $pullRequestNextPageCursor
              states: $pullRequestStates
              orderBy: { field: CREATED_AT, direction: DESC }
            ) {
              totalCount
              pageInfo {
                endCursor
                hasNextPage
              }
              nodes {
                id
                number
                url
                title
                body
                state
                createdAt
                author {
                  login
                  url
                }
                labels(first: $pullRequestPerPage) {
                  nodes {
                    name
                    url
                  }
                }
                timelineItems(first: 10) {
                    nodes { 
                        ... on CrossReferencedEvent {
                            source{
                                ... on Issue{
                                    id
                                    number
                                }
                                ... on PullRequest{
                                    id
                                    number
                                }
                            }
                        }
                        ... on MergedEvent { # merged commit
                            commit {
                                oid,
                                committedDate,
                                parents(first:2){
                                    nodes {
                                        oid,
                                        committedDate,
                                    }
                                }
                            }
                        }
                    }
                }
                closingIssuesReferences(first: $pullRequestPerPage) {
                  nodes {
                    id
                    number
                  }
                }
                commits(first: $pullRequestPerPage) {
                  nodes {
                    commit {
                        oid,
                        committedDate,
                        parents(first:2) {
                            nodes {
                                oid,
                                committedDate,
                            }
                        }
                    }
                  }
                }
                comments(first: $pullRequestPerPage) {
                  totalCount
                  pageInfo {
                    endCursor
                    hasNextPage
                  }
                  nodes {
                    body
                    createdAt
                    createdViaEmail
                    url
                    author {
                      login
                      url
                    }
                  }
                }
              }
            }
          }
        }
    """

    _NODE_COMMENT_QUERY = """
        query($perPage: Int!, $id: ID!, $commentCursor: String!) {
          rateLimit {
            limit
            cost
            remaining
            resetAt
          }
          node(id: $id) {
            ... on Issue {
              comments(first: $perPage, after: $commentCursor) {
                totalCount
                pageInfo {
                  endCursor
                  hasNextPage
                }
                nodes {
                  body
                  createdAt
                  url
                  author {
                    login
                    url
                  }
                }
              }
            }
            ... on PullRequest {
              comments(first: $perPage, after: $commentCursor) {
                totalCount
                pageInfo {
                  endCursor
                  hasNextPage
                }
                nodes {
                  body
                  createdAt
                  url
                  author {
                    login
                    url
                  }
                }
              }
            }
          }
        }
    """

    def __post_init__(self):
        self._session = None  # Requests session
        self._total_pages_fetched = 0
        if not self.token:
            print(
                "No Github access token found, exiting. Use gh2md --help so see options for providing a token."
            )
            sys.exit(1)

        # For testing
        per_page_override = os.environ.get("_GH2MD_PER_PAGE_OVERRIDE", None)
        if per_page_override:
            self.per_page = int(per_page_override)

    def _request_session(self) -> requests.Session:
        if not self._session:
            self._session = requests.Session()
            self._session.headers.update({"Authorization": "token " + self.token})
        return self._session

    def _post(
        self, json: Dict[str, Any], headers: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Make a graphql request and handle errors/retries.
        """
        if headers is None:
            headers = {}
        err = False
        for attempt in range(1, 3):
            try:
                resp = self._request_session().post(
                    self._ENDPOINT, json=json, headers=headers
                )
                resp.raise_for_status()
                err = False
                self._total_pages_fetched += 1
                break
            except Exception:  # Could catch cases that aren't retryable, but I don't think it's too annoying
                err = True
                logger.warning(
                    f"Exception response from request attempt {attempt}", exc_info=True
                )
                time.sleep(3)

        if err:
            decoded = {}
            logger.error("Request failed multiple retries, returning empty data")
        elif int(resp.headers["x-ratelimit-remaining"]) <= 1: #< 4800
            delta = int(resp.headers["x-ratelimit-reset"]) - datetime.datetime.now(timezone.utc).timestamp() + 30
            logger.warning(f"Rate limit exceeded, sleeping {delta} seconds (= {delta/60/60} hours)")
            time.sleep(delta)
            return self._post(json, headers)
        else:
            decoded = resp.json()
            rl = decoded.get("data", {}).get("rateLimit")
            if rl:
                logger.info(
                    f"Rate limit info after request: limit={rl['limit']}, cost={rl['cost']}, remaining={rl['remaining']}, resetAt={rl['resetAt']}"
                )

            errors = decoded.get("errors")
            if errors:
                err = True
                logger.error(f"Found GraphQL errors in response data: {errors}")

        return decoded, err

    def _fetch_repo(
        self,
        owner: str,
        repo: str,
        include_issues: bool,
        include_closed_issues: bool,
        include_prs: bool,
        include_closed_prs: bool,
    ) -> Dict[str, Any]:
        """
        Makes the appropriate number of requests to retrieve all the requested
        issues and PRs.

        Any additional comments beyond the first page in each issue/PR have to
        be fetched separately and merged with the results at the end. This is
        because they live underneath the issue/PR with their own respective
        pagination.
        """

        variables = {
            "owner": owner,
            "repo": repo,
        }
        if include_issues:
            variables["issuePerPage"] = self.per_page
            if not include_closed_issues:
                variables["issueStates"] = ["OPEN"]
        else:
            variables["issuePerPage"] = 0

        if include_prs:
            variables["pullRequestPerPage"] = self.per_page
            if not include_closed_prs:
                variables["pullRequestStates"] = ["OPEN"]
        else:
            variables["pullRequestPerPage"] = 0

        issue_cursor, has_issue_page = None, True
        pr_cursor, has_pr_page = None, True
        success_responses = []
        was_interrupted = False
        while has_issue_page or has_pr_page:
            logger.error(f"cursors: {issue_cursor} {pr_cursor}")
            try:
                # Make the request
                if issue_cursor:
                    variables["issueNextPageCursor"] = issue_cursor
                if pr_cursor:
                    variables["pullRequestNextPageCursor"] = pr_cursor
                data, err = self._post(
                    json={"query": self._REPO_QUERY, "variables": variables}
                )
                if err:
                    break
                else:
                    success_responses.append(data)

                    issues = data["data"]["repository"]["issues"]
                    if issues["nodes"]:
                        issue_cursor = issues["pageInfo"]["endCursor"]
                        has_issue_page = issues["pageInfo"]["hasNextPage"]
                    else:
                        issue_cursor, has_issue_page = None, False

                    prs = data["data"]["repository"]["pullRequests"]
                    if prs["nodes"]:
                        pr_cursor = prs["pageInfo"]["endCursor"]
                        has_pr_page = prs["pageInfo"]["hasNextPage"]
                    else:
                        pr_cursor, has_pr_page = None, False

                    logger.info(
                        f"Fetched repo page. total_requests_made={self._total_pages_fetched}, repo_issue_count={issues['totalCount']}, repo_pr_count={prs['totalCount']} issue_cursor={issue_cursor or '-'} pr_cursor={pr_cursor or '-'}"
                    )
            except (SystemExit, KeyboardInterrupt):
                logger.warning("Interrupted, will convert retrieved data and exit")
                was_interrupted = True
                break

        # Merge all the pages (including comments) into one big response object
        # by extending the list of nodes in the first page. This makes it easier
        # for the rest of the code to deal with it rather than passing around
        # one page at a time. The size of the response data is small enough that
        # memory shouldn't be a concern.
        merged_pages = success_responses[0] if success_responses else {}
        for page in success_responses[1:]:
            merged_pages["data"]["repository"]["issues"]["nodes"].extend(
                page["data"]["repository"]["issues"]["nodes"]
            )
            merged_pages["data"]["repository"]["pullRequests"]["nodes"].extend(
                page["data"]["repository"]["pullRequests"]["nodes"]
            )
        if not was_interrupted:
            self._fetch_and_merge_comments(merged_pages)
        return merged_pages

    def _fetch_and_merge_comments(self, merged_pages: Dict[str, Any]) -> None:
        """
        For any issues/PRs that are found to have an additional page of comments
        available, fetch the comments and merge them with the original data.
        """
        if not merged_pages.get("data"):
            return

        all_nodes = (
            merged_pages["data"]["repository"]["issues"]["nodes"]
            + merged_pages["data"]["repository"]["pullRequests"]["nodes"]
        )

        for original_node in all_nodes:
            if not original_node["comments"]["pageInfo"]["hasNextPage"]:
                continue

            has_page, comment_cursor = (
                True,
                original_node["comments"]["pageInfo"]["endCursor"],
            )
            while has_page:
                try:
                    variables = {
                        "id": original_node["id"],
                        "perPage": self.per_page,
                        "commentCursor": comment_cursor,
                    }
                    data, err = self._post(
                        json={"query": self._NODE_COMMENT_QUERY, "variables": variables}
                    )
                    if err:
                        break
                    else:
                        comments = data["data"]["node"]["comments"]
                        if comments["nodes"]:
                            comment_cursor = comments["pageInfo"]["endCursor"]
                            has_page = comments["pageInfo"]["hasNextPage"]
                        else:
                            comment_cursor, has_page = None, False

                        logger.info(
                            f"Fetched page for additional comments. total_requests_made={self._total_pages_fetched}, issue_comment_count={comments['totalCount']}, comment_cusor={comment_cursor}"
                        )

                        # Merge these comments to the original data
                        original_node["comments"]["nodes"].extend(comments["nodes"])

                except (SystemExit, KeyboardInterrupt):
                    logger.warning("Interrupted, will convert retrieved data and exit")
                    break

    def fetch_and_decode_repository(
        self,
        repo_name: str,
        include_issues: bool,
        include_prs: bool,
        include_closed_issues: bool,
        include_closed_prs: bool,
    ) -> GithubRepo:
        """
        Entry point for fetching a repo.
        """
        logger.info(f"Initiating fetch for repo: {repo_name}")
        owner, repo = repo_name.split("/")
        response = self._fetch_repo(
            owner=owner,
            repo=repo,
            include_issues=include_issues,
            include_prs=include_prs,
            include_closed_issues=include_closed_issues,
            include_closed_prs=include_closed_prs,
        )

        try:
            repo_data = response["data"]["repository"]
        except KeyError:
            logger.error("Repository data missing in response, can't proceed")
            raise

        issues = []
        prs = []

        for i in repo_data["issues"]["nodes"]:
            try:
                issues.append(
                    self._parse_issue_or_pull_request(i, is_pull_request=False)
                )
            except Exception:
                logger.warning(f"Error parsing issue, skipping: {i}", exc_info=True)

        for pr in repo_data["pullRequests"]["nodes"]:
            try:
                prs.append(self._parse_issue_or_pull_request(pr, is_pull_request=True))
            except Exception:
                logger.warning(
                    f"Error parsing pull request, skipping: {pr}", exc_info=True
                )

        return GithubRepo(
            full_name=repo_data["nameWithOwner"],
            url=repo_data["url"],
            # We have to sort in application code because these are separate objects in the GraphQL API.
            issues=sorted(issues + prs, key=lambda x: x.number, reverse=True),
        )

    def _parse_commit(
        self, commits
    )-> List[GithubCommit]:
        githubcommits = []
        for commit in commits:
            id = commit["oid"]
            committed_date = dateutil_parse(commit["committedDate"])
            associatedPullRequests_ids = [node['id'] for node in commit["associatedPullRequests"]["nodes"] if node != {}]
            associatedPullRequests_numbers = [node['number'] for node in commit["associatedPullRequests"]["nodes"] if node != {}]
            parents_ids = [node["oid"] for node in commit["parents"]["nodes"]] if commit["parents"] else []

            githubcommits.append(
                GithubCommit(
                    id = id,
                    committed_date = committed_date,
                    associatedPullRequests_ids = associatedPullRequests_ids,
                    associatedPullRequests_numbers = associatedPullRequests_numbers,
                    parents_ids= parents_ids
                )
            )
        return githubcommits

    def _parse_issue_or_pull_request(
        self, issue_or_pr: Dict[str, Any], is_pull_request: bool
    ) -> GithubIssue:
        i = issue_or_pr
        comments = []
        for c in i["comments"]["nodes"]:
            try:
                comments.append(
                    GithubComment(
                        created_at=dateutil_parse(c["createdAt"]),
                        created_via_email = c["createdViaEmail"] if not (c.get('createdViaEmail') is None) else False,
                        body=c["body"],
                        user_login=c["author"]["login"] if c.get("author") else "(unknown)",
                        user_url=c["author"]["url"] if c.get("author") else "(unknown)",
                        url=c["url"],
                    )
                )
            except Exception:
                logger.warning(f"Error parsing comment, skipping: {c}", exc_info=True)
        
        # if is_pull_request:
        merged_commits_ids = [node["commit"]["oid"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('commit')] if is_pull_request else []

        parent_nodes = {}

        if is_pull_request:
            if len(merged_commits_ids) != 0: # merged commit is supposed to be only one
                parent_nodes = [node["commit"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('commit')][0]
                parent_nodes = parent_nodes["parents"]
            elif len(i["commits"]["nodes"])> 0: 
                #len(merged_commits_ids) == 0: it's risky, eventhough there exist commits associated with the pr, it does not exist locally
                # it is also possible that there are no commmits associated with the pr
                parent_nodes = i["commits"]["nodes"][0]
                parent_nodes = parent_nodes["commit"]["parents"]

        return GithubIssue(
            pull_request=is_pull_request,
            user_login=i["author"]["login"] if i.get("author") else "(unknown)",
            user_url=i["author"]["url"] if i.get("author") else "(unknown)",
            state=i["state"].lower(),
            body=i["body"],
            number=i["number"], 
            title=i["title"],
            created_at=dateutil_parse(i["createdAt"]),
            url=i["url"],
            label_names=[node["name"] for node in i["labels"]["nodes"]],
            closingIssuesReferences_numbers=[node["number"] for node in i["closingIssuesReferences"]["nodes"]] if is_pull_request else [],
            closingIssuesReferences_ids=[node["id"] for node in i["closingIssuesReferences"]["nodes"]] if is_pull_request else [],
            pr_commits_ids=[node["commit"]["oid"] for node in i["commits"]["nodes"]] if is_pull_request else [],
            commits_dates = [node["commit"]["committedDate"] for node in i["commits"]["nodes"]] if is_pull_request else [],
            parents_ids=[node["oid"] for node in parent_nodes["nodes"]] if is_pull_request and parent_nodes !={} else [],
            comments=comments,
            merged_commits_ids = [node["commit"]["oid"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('commit')] if is_pull_request else [],
            id = i['id'],
            sourceIssuesReferences_numbers = [node["source"]["number"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('source')],  
            sourceIssuesReferences_ids = [node["source"]["id"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('source')], 
            associatedPullRequests_numbers = [n["number"] for r in [node["closer"]["associatedPullRequests"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('closer')] for n in r["nodes"] if n not in [None, {}]] \
                                             + [n["number"] for node in i["timelineItems"]["nodes"]if node not in [None, {}] and node.get('commit') for n in node["commit"]["associatedPullRequests"]["nodes"] if n not in [None, {}]] if not is_pull_request else [],
            associatedPullRequests_ids = [n["id"] for r in [node["closer"]["associatedPullRequests"] for node in i["timelineItems"]["nodes"] if node not in [None, {}] and node.get('closer')] for n in r["nodes"] if n not in [None, {}]] \
                                             + [n["id"] for node in i["timelineItems"]["nodes"]if node not in [None, {}] and node.get('commit') for n in node["commit"]["associatedPullRequests"]["nodes"] if n not in [None, {}]] if not is_pull_request else [],
            trackedIssues_ids=[node["id"] for node in i["trackedIssues"]["nodes"]] if not is_pull_request else [],
            trackedIssues_numbers=[node["number"] for node in i["trackedIssues"]["nodes"]] if not is_pull_request else [],
            trackedInIssues_ids=[node["id"] for node in i["trackedInIssues"]["nodes"]] if not is_pull_request else [],
            trackedInIssues_numbers=[node["number"] for node in i["trackedInIssues"]["nodes"]] if not is_pull_request else [],
            closedByPullRequestsReferences_ids=[node["id"] for node in i["closedByPullRequestsReferences"]["nodes"]] if not is_pull_request else [],
            closedByPullRequestsReferences_numbers=[node["number"] for node in i["closedByPullRequestsReferences"]["nodes"]] if not is_pull_request else [],
        )

def export_issues_to_csv(
    repo: GithubRepo,
    output_file_name: str,
) -> None:

    headers = ['number', 'id', 'type', 'state', 'pr_commits', 'merged_commits', 'parents', 'sourceIssuesReferences_numbers', 'closingIssuesReferences_numbers', \
                'trackedIssues_numbers', 'trackedInIssues_numbers', 'closedByPullRequestsReferences_numbers', 'associatedPullRequests_numbers', \
                'sourceIssuesReferences_ids', 'closingIssuesReferences_ids', 'trackedIssues_ids', 'trackedInIssues_ids', 'closedByPullRequestsReferences_ids', 'associatedPullRequests_ids', \
                'labels', 'comments']
    fileName = output_file_name 
   
    with open(fileName, 'w', newline='') as file:
        writer = csv.DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=headers)
        writer.writeheader()

        for pr_or_issue in repo.issues:
            comments = format_issue_to_csv(pr_or_issue)
            type = 'pr' if pr_or_issue.pull_request else 'issue'

            sourceIssuesReferences_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.sourceIssuesReferences_numbers])   
            sourceIssuesReferences_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.sourceIssuesReferences_ids])   
            trackedInIssues_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.trackedInIssues_numbers])   
            trackedInIssues_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.trackedInIssues_ids])   
            trackedIssues_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.trackedIssues_numbers])   
            trackedIssues_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.trackedIssues_ids])   
            closedByPullRequestsReferences_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.closedByPullRequestsReferences_numbers]) 
            closedByPullRequestsReferences_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.closedByPullRequestsReferences_ids]) 
            closingIssuesReferences_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.closingIssuesReferences_numbers]) 
            closingIssuesReferences_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.closingIssuesReferences_ids]) 
            associatedPullRequests_numbers = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.associatedPullRequests_numbers]) 
            associatedPullRequests_ids = ", ".join(["#{}".format(str(number)) for number in pr_or_issue.associatedPullRequests_ids]) 

            pr_commits_str = ", ".join(["{}".format(str(comnum)) for comnum in pr_or_issue.pr_commits_ids])

            merged_commits_str = ", ".join(["{}".format(str(comnum)) for comnum in pr_or_issue.merged_commits_ids])

            labels_names = ", ".join(["{}".format(str(name)) for name in pr_or_issue.label_names])

            parents = ", ".join(["{}".format(str(parnum)) for parnum in pr_or_issue.parents_ids])

            writer.writerow({'number': pr_or_issue.number, 'id': pr_or_issue.id, 'type': type, 'state': str(pr_or_issue.state), 'pr_commits': str(pr_commits_str), \
                'merged_commits': str(merged_commits_str),  \
                'parents': str(parents), \
                'sourceIssuesReferences_numbers': str(sourceIssuesReferences_numbers), \
                'closingIssuesReferences_numbers': str(closingIssuesReferences_numbers), \
                'trackedIssues_numbers': str(trackedIssues_numbers), \
                'trackedInIssues_numbers': str(trackedInIssues_numbers), \
                'closedByPullRequestsReferences_numbers': str(closedByPullRequestsReferences_numbers), \
                'associatedPullRequests_numbers': str(associatedPullRequests_numbers), \
                'sourceIssuesReferences_ids': str(sourceIssuesReferences_ids), \
                'closingIssuesReferences_ids': str(closingIssuesReferences_ids), \
                'trackedIssues_ids': str(trackedIssues_ids), \
                'trackedInIssues_ids': str(trackedInIssues_ids), \
                'closedByPullRequestsReferences_ids': str(closedByPullRequestsReferences_ids), \
                'associatedPullRequests_ids': str(associatedPullRequests_ids), \
                'labels': str(labels_names), \
                'comments': comments})

    return None

def parse_email(email):
    reply = EmailReplyParser.parse_reply(email)
    return reply

def format_issue_to_csv(issue: GithubIssue) -> str:
    """
    Format issue comments as text for CSV export.
    """
    formatted_comments = ""
    if issue.comments:
        comments = []
        for comment in issue.comments:
            if comment.created_via_email:
                body = parse_email(comment.body)
            else:
                body = comment.body
            comments.append(body)
        formatted_comments = "\n\n".join(comments)
    return formatted_comments

def get_environment_token() -> str:
    try:
        logger.info(f"Looking for token in envvar {ENV_GITHUB_TOKEN}")
        token = os.environ[ENV_GITHUB_TOKEN]
        logger.info("Using token from environment")
        return token
    except KeyError:
        for path in GITHUB_ACCESS_TOKEN_PATHS:
            logger.info(f"Looking for token in file: {path}")
            if os.path.exists(path):
                logger.info(f"Using token from file: {path}")
                with open(path, "r") as f:
                    token = f.read().strip()
                    return token

def main():
    """Entry point"""
    args = parse_args(sys.argv[1:])

    gh = GithubAPI(token=get_environment_token())

    repo = gh.fetch_and_decode_repository(
        args.repo,
        include_closed_prs=True,
        include_closed_issues=True,
        include_prs=args.include_prs,
        include_issues=args.include_issues,
    )

    # Log issue counts
    logger.info(f"Retrieved issues for repo: {repo.full_name}")
    counts = {
        "PRs": defaultdict(int),
        "issues": defaultdict(int),
        "total": len(repo.issues),
    }
    for issue in repo.issues:
        if issue.pull_request:
            counts["PRs"][issue.state] += 1
            counts["PRs"]["total"] += 1
        else:
            counts["issues"][issue.state] += 1
            counts["issues"]["total"] += 1
    counts["PRs"] = dict(counts["PRs"])
    counts["issues"] = dict(counts["issues"])
    logger.info(f"Retrieved issue counts: \n{pprint.pformat(counts)}")

    # download git repository locally and then convert and export to csv 
    git_url = 'https://github.com/' + args.repo
    local_dir = args.repo.split('/')[-1]

    if not os.path.exists(local_dir):
        logger.info("Downloding github repository locally")
        local_repo = git.Repo.clone_from(git_url, local_dir)
    else:
        logger.info("Updating github repository locally")
        local_repo = git.Repo(local_dir)
        local_repo.remotes[0].pull()

    logger.info("Converting retrieved issues to csv")

    export_issues_to_csv(
        repo = repo,
        output_file_name =args.output_file_name,
    )
    logger.info("Local github cloning done.")

if __name__ == "__main__":
    sys.exit(main())
