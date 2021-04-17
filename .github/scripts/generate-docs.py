import os
import subprocess

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

    out_file = open('docs.md', 'w')
    out_file.write('# Generated Docs 📜\n')
    out_file.write('⚠️ Do not modify this file because it will be overwritten automatically\n')

    for file in files:
        print(file)
        capture = subprocess.run(['python', file, '--help'], stdout=subprocess.PIPE)

        out_file.write('## ' + file + '\n')
        out_file.write('```\n')
        out_file.write(capture.stdout.decode('UTF-8'))
        out_file.write('```\n')


    print("Done!")

if __name__ == "__main__":
    main()
