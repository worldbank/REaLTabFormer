class SampleEmptyError(Exception):
    """Exception raised for generated samples without valid observations.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(
        self, message="Generated sample is empty after validation.", in_size=None
    ):
        self.in_size = in_size
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.in_size} -> {self.message}"


class SampleEmptyLimitError(SampleEmptyError):
    """Exception raised when SampleEmptyError is raised
    continuously for some specific limit."""

    def __init__(
        self,
        message="Generated sample is still empty after the set limit.",
        in_size=None,
    ):
        super().__init__(message, in_size)
