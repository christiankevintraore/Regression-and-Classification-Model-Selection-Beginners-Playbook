"""online_file_manager.py
~~~~~~~~~~~~~~

Helper for online files management.

Desirable features :

"""



#### Main OnlineFileManager class
class OnlineFileManager:

    def __init__(self, url:str):
        """Initialize the current OnlineFileManager object with required variables.

        """
        self.url = url
        self.fileContent = None

    def get_file_content(self):
        from urllib.request import urlopen

        if self.fileContent is None:
            self.fileContent = urlopen(self.url).read().decode('utf-8')

        return self.fileContent
    
    def read_lines(self, nbLines):
        result = []
        retrievedLines = 0
        currentLine = ''
        for char in self.get_file_content():
            currentLine = currentLine + char

            if char == '\n':
                result.append(currentLine)
                retrievedLines += 1
                currentLine = ''
            
            if retrievedLines == nbLines:
                break

        return result

    def get_input_stream(self):
        from io import StringIO
        return StringIO(self.get_file_content())
