from typing import List


class DataLoader:
    def __init__(self, input_filename):
        self.input_filename = input_filename

    def load(self) -> List[str]:
        lines = []
        with open(self.input_filename, 'r') as input_file:
            for line in input_file:
                lines.append(line)

        return lines
