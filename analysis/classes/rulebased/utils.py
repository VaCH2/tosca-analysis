import os

def get_yaml_files(path):
    '''Get the paths for all the yaml files'''

    extensions = ['.yaml', '.yml']
    allFiles = []

    listOfFile = os.listdir(path)

    for entry in listOfFile:
        fullPath = os.path.join(path, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_yaml_files(fullPath)
        else: 
            for extension in extensions:
                if fullPath.endswith(extension):
                    allFiles.append(fullPath)       
    
    return allFiles


def keyValueList(d): 
    """ 
    This function iterates over all the key-value pairs of a dictionary and returns a list of tuple (key, value).
    d -- a dictionary to iterate through
    """
    if not isinstance(d, dict) and not isinstance(d, list):
        return []

    keyvalues = []

    if isinstance(d, list):
        for entry in d:
            if isinstance(entry, dict):
                keyvalues.extend(keyValueList(entry))
    else:
        for k, v in d.items():
            if k is None or v is None:
                continue
            
            keyvalues.append((k, v))
            keyvalues.extend(keyValueList(v))
                
    return keyvalues


def calculate_depth(f):
    '''https://stackoverflow.com/questions/45964731/how-to-parse-hierarchy-based-on-indents-with-python'''
    
    indentation = []
    indentation.append(0)
    depth = 0

    results = []

    for line in f:
        line = line[:-1]

        content = line.strip()
        indent = len(line) - len(content)
        if indent > indentation[-1]:
            depth += 1
            indentation.append(indent)

        elif indent < indentation[-1]:
            while indent < indentation[-1]:
                depth -= 1
                indentation.pop()

            # if indent != indentation[-1]:
            #     raise RuntimeError("Bad formatting")
        
        results.append((content, depth))
    return results