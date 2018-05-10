

class TSP:
    def __init__(self,filename):
        with open(filename,'r') as inFile:
            table = inFile.readlines()
        table = [row.split(',') for row in table]
        for row in table:
            for i in range(len(row)):
                row[i] = row[i].strip()
                if row[i].replace('.', '', 1).isdigit():
                    row[i]=float(row[i])

        self.TSPmatrix = table


    def getMatrix(self):
        return self.TSPmatrix







