from lxml import etree
import sys
import os
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse(infile, outfile, repo_path):
    """Parse XML file and simplify by removing unnecessary elements and attributes."""
    try:
        parser = etree.XMLParser(remove_comments=False)
        tree = etree.parse(infile, parser=parser)
        root = tree.getroot()
        
        # Normalize RFC version 3 to version 2
        for name, value in root.attrib.items():
            if name == "version" and value == "3":
                root.attrib["version"] = "2"
        
        # Remove consensus attribute if present
        if 'consensus' in root.attrib:
            del root.attrib['consensus']
        
        # Get main RFC sections
        front = root.find('front')
        middle = root.find('middle')
        back = root.find('back')
        
        # Process front section
        if front is not None:
            remove(front, 'xref', 't', True)
        
        # Process middle section
        if middle is not None:
            # Remove acknowledgement(s) sections (case-insensitive, singular or plural)
            for section in list(middle.findall("./section")):
                title_lower = section.attrib.get('title', '').lower()
                if title_lower in ('acknowledgement', 'acknowledgements'):
                    middle.remove(section)
            
            # Remove tables from IANA section (case-insensitive, singular or plural)
            iana = next((section for section in middle.findall("./section")
                        if section.attrib.get('title', '').lower() in ('iana consideration', 'iana considerations')), None)
            if iana is not None:
                for texttable in list(iana.iter('texttable')):
                    texttable.getparent().remove(texttable)
            
            # Clear section titles
            for section in middle.iter('section'):
                section.attrib["title"] = ""
            
            # Remove optional elements (figures, tables, code, etc.)
            for element_type in ['figure', 'texttable', 'dl', 'sourcecode', 'artwork']:
                for elem in list(middle.iter(element_type)):
                    elem.getparent().remove(elem)
            
            # Remove contributors section
            contributors = middle.find("./section[@anchor='contributors']")
            if contributors is not None:
                middle.remove(contributors)
            
            # Remove comment references
            for cref in list(middle.iter('cref')):
                cref.getparent().remove(cref)
            
            # Remove references from text
            remove(middle, 'bcp14', 't', True)
            remove(middle, 'xref', 't', True)
            remove(middle, 'xref', 'li', True)
            remove(middle, 'relref', 't', True)
            remove(middle, 'relref', 'li', True)
            
            # Clear list styles
            for lst in middle.iter('list'):
                for key in list(lst.attrib.keys()):
                    lst.attrib[key] = ""

        
        # Remove back section (references, etc.)
        if back is not None:
            root.remove(back)
        
        # Write simplified XML
        tree.write(outfile)
        logger.info(f"Processed: {infile} -> {outfile}")
        
    except Exception as e:
        logger.error(f"Error processing {infile}: {e}")

def remove(root, remove_tag, search_tag='t', keep=False, attr_name='target'):
    """Remove elements while preserving text content."""
    for node in root.iter(search_tag):
        prev = None
        for child in list(node):
            if child.tag == remove_tag:
                # Preserve child node's tail text
                if child.tail:
                    if prev is None:
                        if keep:
                            # Keep element text or attribute value
                            if remove_tag == 'bcp14':
                                node.text = (node.text or '') + (child.text or '') + child.tail
                            else:
                                node.text = (node.text or '') + \
                                    (child.attrib.get(attr_name, '') or '') + child.tail
                        else:
                            node.text = (node.text or '') + child.tail
                    else:
                        if keep:
                            if remove_tag == 'bcp14':
                                prev.tail = (prev.tail or '') + (child.text or '') + child.tail
                            else:
                                prev.tail = (prev.tail or '') + \
                                    (child.attrib.get(attr_name, '') or '') + child.tail
                        else:
                            prev.tail = (prev.tail or '') + child.tail
                node.remove(child)
            else:
                prev = child

def main():
    """Main entry point for simplifying XML files."""
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <repo_path>")
    
    repo_path = sys.argv[1]
    snapshots_dir = os.path.join(repo_path, "snapshots")
    
    if not os.path.exists(snapshots_dir):
        logger.error(f"Snapshots directory not found: {snapshots_dir}")
        return
    
    logger.info(f"Processing XML files in {snapshots_dir}")
    
    output_dir = os.path.join(repo_path, "simplified_xmls")
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    for infile in glob.glob(os.path.join(snapshots_dir, "*.xml")):
        file_name = os.path.splitext(os.path.basename(infile))[0]
        outfile = os.path.join(output_dir, f"{file_name}.out.xml")
        parse(infile, outfile, repo_path)
        processed += 1
    
    logger.info(f"Completed: processed {processed} XML files")

if __name__ == "__main__":
    main()

