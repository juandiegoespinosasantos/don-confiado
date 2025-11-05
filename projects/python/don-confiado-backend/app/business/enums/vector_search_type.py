from enum import Enum

class VectorSearchType(str, Enum):

    def __new__(cls, value: str, operator: str):
        member = str.__new__(cls, value)
        member._value_ = value
        member.operator = operator

        return member

    EUCLIDEAN_DISTANCE = ("EUCLIDEAN", "<->")
    COSINE_DISTANCE = ("COSINE", "<=>")
    DOT_PRODUCT_DISTANCE = ("DOT_PRODUCT", "<#>")