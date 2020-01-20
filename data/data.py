import os

class Data():
    def __init__(self):
        pass

    def get_yaml_files(self, path):
        extensions = ['.yaml', '.yml']
        allFiles = []

        listOfFile = os.listdir(path)

        for entry in listOfFile:
            fullPath = os.path.join(path, entry)
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.get_yaml_files(fullPath)
            else: 
                for extension in extensions:
                    if fullPath.endswith(extension):
                        allFiles.append(fullPath)       
        
        return allFiles

    