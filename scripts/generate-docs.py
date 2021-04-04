import os

def filter_py_files(value):
    if (value.split('.')[-1] == 'py'):
        return True
    return False

def main():
    print("💫 Generating Docs ... 💫")

    # check run dir
    assert os.getcwd().split('/')[-1] == 'dataset-tools', 'Please run script from dataset-tools root dir.'

    files = os.listdir('./')
    files = list(filter(filter_py_files, files))

    for file in files:
        print(file)
        f = open(file, 'r')
        print(f.read())

    print("Done!")

if __name__ == "__main__":
    main()
