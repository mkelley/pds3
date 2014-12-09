# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
pds3 --- Simple (minded) PDS3 tools
===================================

read_label       - Read a PDS3 label.
read_ascii_table - Read an ASCII table as described by a label.
read_table       - Read a table as described by a label.

"""

__all__ = [
    'read_label',
    'read_ascii_table',
    'read_table'
]

try:
    from ply import lex, yacc
except ImportError:
    raise ImportError("pds3 requires the PLY (Python Lex-Yacc) module.")

class IllegalCharacter(Exception):
    pass

class Parser():
    tokens = ['KEYWORD', 'POINTER', 'STRING', 'INT', 'REAL',
              'UNIT', 'DATE', 'END']

    literals = list('=(),')

    t_POINTER = r'\^[A-Z0-9_]+'
    t_ignore_COMMENT = r'/\*.+?\*/'
    t_ignore = ' \t\r\n'

    # lower case PDS3 to astropy unit translation
    unit_translate = dict(v='V', k='K')

    def __init__(self, debug=False):
        import os
        self.debug = debug
        self.lexer = lex.lex(module=self, debug=self.debug)
        self.parser = yacc.yacc(module=self, debug=self.debug,
                                write_tables=0)

    def parse(self, raw_label, **kwargs):
        return self.parser.parse(raw_label, lexer=self.lexer, debug=self.debug,
                                 **kwargs)

    def t_KEYWORD(self, t):
        r'[A-Z][A-Z0-9_:]+'
        if t.value == 'END':
            t.type = 'END'
        return t

    def t_DATE(self, t):
        r'\d\d\d\d-\d\d-\d\d(T\d\d:\d\d(:\d\d(.\d+)?)?)?Z?'
        from astropy.time import Time
        t.value = Time(t.value, scale='utc')
        return t

    def t_UNIT(self, t):
        r'<[\w*^\-/]+>'
        import astropy.units as u

        # most astropy units are lower-case versions of the PDS3 units
        unit = t.value[1:-1].lower()

        # but not all
        if unit in self.unit_translate:
            unit = self.unit_translate[unit]

        t.value = u.Unit(unit)
        return t

    def t_STRING(self, t):
        r'"[^"]+"'
        t.value = t.value[1:-1].replace('\r', '')
        return t
 
    def t_REAL(self, t):
        r'[+-]?(([0-9]+\.[0-9]*)|(\.[0-9]+))([Ee][+-]?[0-9]+)?'
        t.value = float(t.value)
        return t

    def t_INT(self, t):
        r'[+-]?[0-9]+'
        t.value = int(t.value)
        return t

    def t_error(self, t):
        raise IllegalCharacter(t.value[0])

    def lexer_test(self, data):
        self.lexer.input(data)
        while True:
            tok = self.lexer.token()
            if not tok:
                break
            print tok

    def p_label(self, p):
        """label : record
                 | label record
                 | label END"""
        if len(p) == 2:
            # record
            p[0] = [p[1]]
        elif p[2] == 'END':
            # label END
            p[0] = p[1]
        else:
            # label record
            p[0] = p[1] + [p[2]]

    def p_record(self, p):
        """record : KEYWORD '=' value
                  | POINTER '=' STRING
                  | POINTER '=' '(' STRING ',' INT ')'"""
        if len(p) == 4:
            p[0] = (p[1], p[3])
        else:
            p[0] = (p[1], (p[4], p[6]))

    def p_value(self, p):
        """value : STRING
                 | DATE
                 | KEYWORD
                 | number
                 | quantity
                 | sequence"""
        p[0] = p[1]

    def p_value_quantity(self, p):
        """quantity : number UNIT"""
        p[0] = p[1] * p[2]

    def p_number(self, p):
        """number : INT
                  | REAL"""
        p[0] = p[1]

    def p_sequence(self, p):
        """sequence : '(' value ')'
                    | '(' sequence_values ')'"""
        p[0] = p[2]

    def p_sequence_values(self, p):
        """sequence_values : value ','
                           | sequence_values value ','
                           | sequence_values value"""
        if p[2] == ',':
            p[0] = [p[1]]
        else:
            p[0] = p[1] + [p[2]]

    def p_error(self, p):
        if p:
            print("Syntax error at '%s'" % p.value)
        else:
            print("Syntax error at EOF")


def _records2dict(records, object_index=0):
    """Convert a list of PDS3 records to a dictionary.

    Parameters
    ----------
    records : list
      List of key-item pairs.
    object_index : int, optional
      Extract just a single object or group, starting at the index
      `object_index`.

    Returns
    -------
    d : dict
      The dictionary.
    last_index : int, optional
      When extracting a single object or group, also return the last
      index of that object.

    """

    label = dict()
    start = 0
    if object_index != 0:
        start = object_index
        object_name = records[start][1]
        start += 1
    else:
        object_name = None

    i = start
    while i < len(records):
        # groups and objects are both terminated with 'END_OBJECT'
        if records[i] == ('END_OBJECT', object_name):
            return label, i
        elif records[i][0] in ['OBJECT', 'GROUP']:
            key = records[i][1]
            value, j = _records2dict(records, i)
            i = j
        else:
            key = records[i][0]
            value = records[i][1]

        if key in label:
            if not isinstance(label[key], list):
                label[key] = [label[key]]
            label[key].append(value)
        else:
            label[key] = value
        i += 1

    return label

def _find_file(filename, path='.'):
    """Search a directory for a file.

    PDS3 file names are required to be upper case, but in case-
    sensitive file systems, the PDS3 file name many not match the file
    system file name.  `_find_file` will make a case insentive
    comparison to all files in the given path.

    Parameters
    ----------
    filename : string
      The PDS3 file name for which to search.
    path : string, optional
      Search within this directory path.

    Returns
    -------
    fn : string
      The actual name of the file on the file system, including the
      path.

    Raises
    ------
    IOError
      When the file is not found.
      
    """

    import os
    for f in sorted(os.listdir(path)):
        if f.lower() == filename.lower():
            f = os.path.sep.join([path, f])
            return f
    raise IOError("No match for {} in {}".format(filename, path))

def read_label(filename, debug=False):
    """Read in a PDS3 label.

    Parameters
    ----------
    filename : string
      The name of the file to read.

    Returns
    -------
    label : dict
      The label as a `dict`.

    Raises
    ------
    IllegalCharacter

    Notes
    -----
    Objects and groups are returned as dictionaries containing all
    their sub-keywords.  Multiple objects (e.g., columns) with the
    same name are returned as a `list` of objects.

    """

    with open(filename, 'r') as inf:
        raw_label = inf.read(-1)

    parser = Parser(debug=debug)
    records = parser.parse(raw_label)
    return _records2dict(records)

def read_ascii_table(label, key, path='.'):
    """Read an ASCII table as described by the label.

    Only fixed length records are supported.

    Parameters
    ----------
    label : dict
      The label, as read by `read_label`.
    key : string
      The label key of the object that describes the table.
    path : string, optional
      Directory path to label/table.

    Returns
    -------
    table : astropy Table

    Raises
    ------
    NotImpementedError
    ValueError

    """

    import numpy as np
    from astropy.io import ascii

    # The table object description.
    desc = label[key]

    if not isinstance(desc['COLUMN'], list):
        # For tables with a single column, desc['COLUMN'] needs to be a list
        desc['COLUMN'] = [desc['COLUMN']]

    # Setup table column formats
    n = desc['COLUMNS']
    col_starts = []
    col_ends = []
    converters = dict()
    for i in range(n):
        col = desc['COLUMN'][i]
        col_starts.append(col['START_BYTE'] - 1)
        col_ends.append(col_starts[-1] + col['BYTES'] - 1)

        if col['DATA_TYPE'] == 'ASCII_REAL':
            dtype = np.float
        elif col['DATA_TYPE'] == 'ASCII_INTEGER':
            dtype = np.int
        elif col['DATA_TYPE'] == 'CHARACTER':
            dtype = np.dtype('S{}'.format(col['BYTES']))
        else:
            raise ValueError("Unknown data type: ", col['DATA_TYPE'])
        converters['col{}'.format(i+1)] = [ascii.convert_numpy(dtype)]

    nrows = desc['ROWS']

    # Open the file object, and skip ahead to the start of the table,
    # if needed.  Read the table.
    if isinstance(label['^' + key], tuple):
        filename, start = label['^' + key]
        start = int(start) - 1
        filename = _find_file(filename, path=path)
        if 'RECORD_BYTES' in label:
            record_bytes = label['RECORD_BYTES']
        else:
            record_bytes = desc['RECORD_BYTES']

        #inf = open(filename, 'r')
        #inf.seek(record_bytes * start)
    else:
        filename = _find_file(label['^' + key], path=path)
        start = 0
        #inf = open(filename, 'r')

    table = ascii.read(filename, format='fixed_width_no_header',
                       data_start=start, data_end=nrows+start,
                       col_starts=col_starts, col_ends=col_ends,
                       converters=converters, guess=False)
    #inf.close()

    # Mask data
    for i in range(n):
        col = desc['COLUMN'][i]
        missing_constant = col.get('MISSING_CONSTANT', None)
        if missing_constant is None:
            continue

        j = table.columns[i] == missing_constant
        if np.any(j):
            table.columns[i].mask = j

    # Save column meta data.
    for i in range(n):
        col = desc['COLUMN'][i]
        table.columns[i].name = col['NAME']
        if 'DESCRIPTION' in col:
            table.columns[i].description = col['DESCRIPTION']

    # Save table meta data.
    for k, v in desc.items():
        if k is not 'COLUMN':
            table.meta[k] = v

    return table


def read_table(label, key, path='.'):
    """Read table as described by the label.

    Calls `read_ascii_table` or `read_binary_table` as appropriate.

    Parameters
    ----------
    label : dict
      The label, as read by `read_label`.
    key : string
      The label key of the object that describes the table.
    path : string, optional
      Directory path to label/table.

    Returns
    -------
    table : astropy Table

    """

    format = label[key]['INTERCHANGE_FORMAT']
    if format == 'ASCII':
        return read_ascii_table(label, key, path=path)
    else:
        raise NotImplementedError("Table format not implemented: {}".format(format))
