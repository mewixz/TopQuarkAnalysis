import json
from os import listdir
from os.path import isfile, join
import sys

mypath = sys.argv[1]
extlist = ['png']

all_files = listdir(mypath)

output_json = {}

for f in all_files:
    f_ext = f.split('.')
    if isfile(join(mypath, f)) and f_ext[-1] in extlist:
        parts = f_ext[0].split('_')
        isLog = parts[-1]=='log'
        if not isLog:
            cat = parts[0]
            if len(parts) > 1:
                var = "_".join( parts[1:] )
            else:
                var = parts[0]

            available_formats = {}
            for ext in ['pdf', 'cxx', 'eps', 'root', 'txt']:
                if f_ext[0] + '.' + ext in all_files:
                    available_formats[ ext ] = f_ext[0] + '.' + ext

            LogAvailable = f_ext[0] + "_log." + f_ext[1] in all_files

            
            if not cat in output_json :
                output_json[cat] = {}

            output_json[cat][ var ] = { 'image': f ,
                                        'hasLog': LogAvailable,
                                        'otherFormats': available_formats }
            if LogAvailable:
                output_json[cat][ var ][ 'logImage' ] = f_ext[0] + "_log." + f_ext[1]


#with open('data.json' , 'w') as f:
#    json.dump( output_json , f )


from xml.etree.ElementTree import ElementTree, Element, SubElement, Comment, tostring
root = Element("categories")
for cat in output_json:
    cat_el = SubElement(root, "category")
    cat_name = SubElement( cat_el , 'name').text = cat
    variables = SubElement( cat_el , 'variables')
    for var in output_json[cat]:
        var_el = SubElement(variables, "variable")
        SubElement( var_el , "name" ).text = var
        SubElement( var_el , "image" ).text = output_json[cat][var]['image']
        SubElement( var_el , "HasLog").text = str(output_json[cat][var]['hasLog'])
        SubElement( var_el , "logImage").text = output_json[cat][ var ][ 'logImage' ]
        other_formats = SubElement( var_el , "OtherFormats" )
        for of in output_json[cat][var]['otherFormats']:
            of_el = SubElement( other_formats , "format" )
            SubElement( of_el , "ext" ).text = of
            SubElement( of_el , "file" ).text = output_json[cat][var]['otherFormats'][of]
            
tree = ElementTree(root)
tree.write(mypath+"/data.xml")
    
import xml.dom.minidom
dom = xml.dom.minidom.parse(mypath+'/data.xml')
pi = dom.createProcessingInstruction('xml-stylesheet',
                                     'type="text/xsl" href="/hbakhshi/SMP-19-005/categories.xsl"')
dom.insertBefore(pi, dom.firstChild)
pretty_xml_as_string = dom.toprettyxml()
with open(mypath+'/data.xml' , 'w') as f:
    f.write( pretty_xml_as_string )
