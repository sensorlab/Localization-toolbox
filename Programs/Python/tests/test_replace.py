class test_class(object):

    def __init__(self):
        self.Ids = []
        self.Data = []
        return

    def get_Ids(self):
        return self.Ids

    def append(self, id, data):
        self.Ids.append(id)
        self.Data.append(data)
        return

    def get_index(self, id):
        return self.Ids.index(id)

    def delete(self, Id):
        ix = self.get_index(Id)
        self.Data.pop(ix)
        self.Ids.pop(ix)
        print ix
        return

    def replace(self, Id, data):
        ix = self.get_index(Id)
        self.delete(Id)
        self.append(Id, data)
        return

    def print_data(self):
        for ix, x in zip(self.Ids, self.Data):
            print ix, " ---->  ",x
        return

if __name__ == '__main__':
    print "Test replace"
    list_data = test_class()

    data_in = {"a": "A11", "b": "A12", "c": "A14"}
    list_data.append("A", data_in)

    data_in = {"a": "B11", "b": "B12", "c": "B14"}
    list_data.append("B", data_in)

    data_in = {"a": "C11", "b": "C12", "c": "C14"}
    list_data.append("C", data_in)

    data_in = {"a": "D11", "b": "D12", "c": "D14"}
    list_data.append("D", data_in)

    data_in = {"a": "E11", "b": "E12", "c": "E14"}
    list_data.append("E", data_in)

    list_data.print_data()

    list_data.replace("C", data_in)

    list_data.print_data()
