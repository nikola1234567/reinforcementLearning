from enum import Enum


class FileFormats(Enum):
    csv = 1
    not_supported = 2

    def format_name(self):
        return self.name.replace("-", "")

    def format_description(self):
        descriptions = {FileFormats.csv: "Comma separated value (CSV)",
                        FileFormats.not_supported: 'Not supported file type'}
        return descriptions[self]


if __name__ == '__main__':
    print(FileFormats.csv.format_name())
    print(FileFormats.csv.format_description())
    print(FileFormats.not_supported.format_name())
    print(FileFormats.not_supported.format_description())
