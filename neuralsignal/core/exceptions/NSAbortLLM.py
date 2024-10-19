class NSAbortLLM(Exception):

    def __init__(self, message: str):
        self.value = message
        super().__init__(message)

    def __str__(self):
        return (repr(self.value))
