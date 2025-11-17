BERT_BLOCK_PARTITIONS = [
    r"^bert\.encoder\.layer\.\d+\.attention\.self\.query$",  # Matches all module names with "attention.self.query"
    r"^bert\.encoder\.layer\.\d+\.attention\.self\.key$",  # Matches all module names with "attention.self.key"
    r"^bert\.encoder\.layer\.\d+\.attention\.self\.value$",  # Matches all module names with "attention.self.value"
    r"^bert\.encoder\.layer\.\d+\.attention\.output\.dense$",  # Matches all module names with "attention.output.dense"
    r"^bert\.encoder\.layer\.\d+\.intermediate\.dense$",  # Matches all module names with "intermediate.dense"
    r"^bert\.encoder\.layer\.\d+\.output\.dense$",  # Matches all module names with "output.dense"
]
