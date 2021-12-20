import csv

class CSVHandler:
    def __init__(self, filePath):
        self.filePath = filePath

    def saveRow(self, row):
        with open(self.filePath, 'a', encoding='UTF8') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def saveRows(self, rows):
        with open(self.filePath, 'a', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    def saveDictionary(self, dictionaryKeys, dictionary):
        with open(self.filePath, 'a', encoding='UTF8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=dictionaryKeys)
            writer.writerows(dictionary)

