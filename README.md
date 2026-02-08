# RFCRationaleBuilder

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- For `pygit2` on macOS: `brew install libgit2`
- For `pygit2` on Ubuntu/Debian: `sudo apt-get install libgit2-dev`

### 1. Create a Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv gh2md-venv

# Activate virtual environment
# On macOS/Linux:
source gh2md-venv/bin/activate

# On Windows:
gh2md-venv\Scripts\activate
```

### 2. Install Required Packages

```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# Check that packages are installed
pip list
```

### Note: Using the Virtual Environment

Always activate the virtual environment before running scripts:

```bash
# Activate (macOS/Linux)
source gh2md-venv/bin/activate

# Deactivate when done
deactivate
```

---

## Processing Pipeline

#### Step 1: Download Data from GitHub
```
# gh2md wg_url/draft_name draft_name.csv, e.g.
python gh2csv_ids.py gh2csv_ids.py ietf-wg-add/draft-ietf-add-svcb-dns draft-ietf-add-svcb-dns.csv
```
#### Step 2: Restore File Snapshots
We keep only the xml, markdown files, which are the actual draft files. The restored files are stored in a folder "in".
```
# restore_file.py local_repo_path(the same as draft_name) draft_name(.csv), e.g.
python restore_file.py draft-ietf-add-svcb-dns
```
#### Step 3: Simplify and Convert XML/Markdown Files

```
# depends on the type of file: .md, .mkd or .xml, and if it is a markdown type(.md or .mkd):
for FILE in in/*.md; do kramdown-rfc2629 -2 $FILE > $FILE.xml; done; 
# kramdown-rfc thinks both v2 and v3 are set to v3 by default, we force it to be version 2 by "-2", since v3 cannot remove the paging.
```
There are different formats of markdown files, so is to be processed by mmark, 
draft-ietf-dmarc-dmarcbis, draft-ietf-dnsop-glue-is-not-optional, perc-wg, sniencryption;
```
https://github.com/mmarkdown/mmark
for FILE in draft-ietf-add-svcb-dns/snapshots/*.md; do ./mmark $FILE > $FILE.xml; done; 
```
```
# simplify_xml_parser local_repo_path, e.g.
python simplify_xml_parser.py draft-ietf-add-svcb-dns

 ```
During this step, it need manually fix the xml files, depends on the following:
 ```
# missing closing tag, e.g. <tag/>
# tag mismatch
# missing definition, e.g.
<!DOCTYPE definition [
  <!ENTITY uuml   "&#220;">
  <!ENTITY nbsp   "&#160;">
  <!ENTITY nbhy   "&#8209;">
  <!ENTITY aacute "&#193;">
  <!ENTITY mdash  "&#151;">
  <!ENTITY zwsp   "&#8203;">
  <!ENTITY wj     "&#8288;">
  <!ENTITY rsquo  "&#146;">
  <!ENTITY rdquo  "&#148;">
  <!ENTITY ldquo  "&#147;">
  <!ENTITY iacute "&#205;">
  <!ENTITY eacute "&#201;">
  <!ENTITY wj     "&#8288;">
  <!ENTITY reg    "&#174;">
]>
https://authors.ietf.org/en/templates-and-schemas
# CData section not finished
&#8220;<bcp14></bcp14>&#8221

markdown file corruption correcting tool:
https://codebeautify.org/yaml-validator
xml file corruption correcting tool:
https://codebeautify.org/xmlvalidator
 ```
#### Step 4: Convert XML to Text
```
for FILE in draft-ietf-add-svcb-dns/simplified_xmls/*.out.xml; do xml2rfc $FILE --v2 --raw --no-dtd; done;
```

#### Step 5: Extract Changed Paragraphs and Discussion Text

```
# txt2csv local_repo_path draft_name_ab(.csv), e.g.
python txt2csv_ids.py draft-ietf-add-svcb-dns --no-save-filtered --no-save-duplicates
```