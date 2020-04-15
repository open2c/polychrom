from polychrom.polymerutils import load, save
import os


class LegacyReporter(object):
    def __init__(self, folder):
        """
        
        """
        if not os.path.exists(folder):
            os.mkdir(folder)
        if len(os.listdir(folder)) != 0:
            raise RuntimeError(f"folder {folder} is not empty")
        self.folder = folder
        self.counter = {}

    def report(self, name, values):
        count = self.counter.get(name, 0)

        if name in ["data"]:
            filename = os.path.join(self.folder, "block{0}.dat".format(count))
            save(values["pos"], filename)

        else:
            pass
        self.counter[name] = count + 1

    def dump_data(self):
        pass
