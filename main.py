# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from LabData.DataLoaders.Loader import Loader
from LabData.DataLoaders.BodyMeasuresLoader import BodyMeasuresLoader

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    X = BodyMeasuresLoader().get_data(study_ids=['10k'])
    print("here")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
