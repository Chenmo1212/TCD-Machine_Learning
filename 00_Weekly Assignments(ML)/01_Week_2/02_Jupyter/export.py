import json, re
from typing import Dict


def is_export(cell: Dict) -> bool:
    '''
    use this function to determine whether
    the code in current cell needs to be written to a pyfile.
    '''
    if cell['cell_type'] != 'code': return False
    src = cell['source']
    # import pdb; pdb.set_trace()
    return re.match(r'^\s*#\s*export\s*$', src[0], re.IGNORECASE) is not None


def nbpy2py(fname: str) -> None:
    '''
    parse a nbpy file and convert
    necessary part into a py file with thre same prefix_name.py.
    '''
    fname_result = 'nb_{}.py'.format(fname.split('.')[0])
    # open the file and read it as a dic
    nb_data = json.load(open(fname, 'r', encoding='utf-8'))
    # get all the cells that needs to be exported
    code_cells = [code_cell for code_cell in nb_data['cells'] if is_export(code_cell)]

    py_file_content = ''
    for cell in code_cells:
        py_file_content += ''.join(cell['source'][1:]) + '\n\n'
    # remove the trailing spaces
    py_file_content = re.sub(r'\s+$', '', py_file_content, flags=re.MULTILINE)
    with open(fname_result, 'w', encoding='utf-8') as f:
        f.write(py_file_content)

    print('coverted {} to {}'.format(fname, fname_result))


fname = 'Untitled.ipynb'

nbpy2py(fname)