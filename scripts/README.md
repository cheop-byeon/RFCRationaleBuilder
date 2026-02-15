## Processing Pipeline

### Step 1: Download data from GitHub

This step is based on gh2md. See https://github.com/mattduck/gh2md.

```bash
# Use GraphQL to scrape data and save to CSV
# Usage: wg_url/draft_name draft_name.csv
python gh2csv_ids.py ietf-wg-add/draft-ietf-add-svcb-dns draft-ietf-add-svcb-dns.csv
```

### Step 2: Restore file snapshots

Keep only XML/Markdown draft files. Restored files are stored in the `snapshots/` folder.

```bash
# restore_file.py local_repo_path (same as draft_name)
python restore_file.py draft-ietf-add-svcb-dns
```

### Step 3: Convert Markdown/XML and simplify XML

For kramdown, see https://github.com/gettalong/kramdown.

```bash
# For Markdown inputs (.md/.mkd)
for FILE in snapshots/*.md; do kramdown-rfc2629 -2 "$FILE" > "$FILE".xml; done
# kramdown-rfc defaults to v3; force v2 with "-2" because v3 does not remove paging.
```

Some drafts use mmark (e.g., draft-ietf-dmarc-dmarcbis, draft-ietf-dnsop-glue-is-not-optional, perc-wg, sniencryption).
See https://github.com/mmarkdown/mmark.

```bash
for FILE in draft-ietf-add-svcb-dns/snapshots/*.md; do ./mmark "$FILE" > "$FILE".xml; done
```

```bash
# Simplify XML
python simplify_xml_parser.py draft-ietf-add-svcb-dns
```

During this step, malformed XML may require manual fixes, such as:

- Missing closing tags (e.g., `<tag/>`)
- Tag mismatches
- Missing entity definitions, e.g.

```
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
```

Reference: https://authors.ietf.org/en/templates-and-schemas

- Unfinished CDATA sections (e.g., `&#8220;<bcp14></bcp14>&#8221`)

Validation tools:

- Markdown: https://codebeautify.org/yaml-validator
- XML: https://codebeautify.org/xmlvalidator

### Step 4: Convert XML to text

See https://github.com/ietf-tools/xml2rfc.

```bash
for FILE in draft-ietf-add-svcb-dns/simplified_xmls/*.out.xml; do xml2rfc "$FILE" --v2 --raw --no-dtd; done
```

### Step 5: Extract changed paragraphs and discussion text

```bash
# txt2csv local_repo_path draft_name_ab(.csv)
python txt2csv_ids.py draft-ietf-add-svcb-dns --no-save-filtered --no-save-duplicates
```

## Output Files

Typical outputs include:

- **snapshots/**: restored draft files from repo history.
- **simplified_xmls/**: normalized XML and text derivatives.
- ***.csv**: extracted pairs of changed paragraphs and discussion context.