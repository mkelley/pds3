# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = [
    'read_label',
    'read_ascii_table',
    'read_image',
    'read_table'
]

from warnings import warn
from collections import OrderedDict

try:
    from ply import lex, yacc
except ImportError:
    raise ImportError("pds3 requires the PLY (Python Lex-Yacc) module.")


class ExperimentalFeature(Warning):
    pass


class IllegalCharacter(Exception):
    pass


class PDS3Keyword(str):
    """PDS3 keyword.

    In the following, the keyword is "IMAGE":

      OBJECT = IMAGE
        ...
      END_OBJECT = IMAGE

    """
    def __new__(cls, value):
        return str.__new__(cls, value)


class PDS3Object(OrderedDict):
    """PDS3 data object definition.

    OBJECT = IMAGE
      ...
    END_OBJECT = IMAGE

    """
    pass


class PDS3Group(OrderedDict):
    """PDS3 group statement.

    GROUP = SHUTTER_TIMES
      ...
    END_GROUP = SHUTTER_TIMES

    """
    pass


PDS3_DATA_TYPE_TO_DTYPE = {
    'CHARACTER': 'a',
    'IEEE_REAL': '>f',
    'LSB_INTEGER': '<i',
    'LSB_UNSIGNED_INTEGER': '<u',
    'MAC_INTEGER': '>i',
    'MAC_REAL': '>f',
    'MAC_UNSIGNED_INTEGER': '>u',
    'MSB_UNSIGNED_INTEGER': '>u',
    'MSB_INTEGER': '>i',
    'PC_INTEGER': '<i',
    'PC_UNSIGNED_INTEGER': '<u',
    'SUN_INTEGER': '>i',
    'SUN_REAL': '>f',
    'SUN_UNSIGNED_INTEGER': '>u',
    'VAX_INTEGER': '<i',
    'VAX_UNSIGNED_INTEGER': '<u',
}


class PDS3Parser():
    tokens = ['KEYWORD', 'POINTER', 'STRING', 'INT', 'REAL',
              'UNIT', 'DATE', 'END']

    literals = list('=(){},')

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
        r'[A-Z][A-Z0-9_:]*'
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
            print(tok)

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
                  | POINTER '=' INT
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
                 | pds_set
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

    def p_pds_set(self, p):
        """pds_set : '{' value '}'
                   | '{' sequence_values '}'"""
        p[0] = set(p[2])

    def p_sequence(self, p):
        """sequence : '(' value ')'
                    | '(' sequence_values ')'
                    | '{' value '}'
                    | '{' sequence_values '}'"""
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

    from collections import OrderedDict

    label = OrderedDict()
    start = 0
    if object_index != 0:
        start = object_index
        object_name = records[start][1]
        start += 1
    else:
        object_name = None

    i = start
    while i < len(records):
        # groups and objects are both terminated with 'END_...'
        if (records[i] == ('END_OBJECT', object_name)
                or records[i] == ('END_GROUP', object_name)):
            return label, i
        elif records[i][0] in ['OBJECT', 'GROUP']:
            key = PDS3Keyword(records[i][1])
            value, j = _records2dict(records, i)
            if records[i][0] == 'OBJECT':
                value = PDS3Object(value)
            elif records[i][0] == 'GROUP':
                value = PDS3Group(value)
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

    raw_label = ''
    with open(filename, 'rb') as inf:
        while True:
            line = inf.readline()
            raw_label += line.decode('ascii')
            if line.strip() == b'END' or line == b'':
                break

    parser = PDS3Parser(debug=debug)
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

    def repeat(dtype, col):
        n = col.get('ITEMS', 1)
        return dtype if n == 1 else (dtype,) * n

    for i in range(n):
        col = desc['COLUMN'][i]
        col_starts.append(col['START_BYTE'] - 1)
        col_ends.append(col_starts[-1] + col['BYTES'] - 1)

        if col['DATA_TYPE'] == 'ASCII_REAL':
            dtype = repeat(np.float, col)
        elif col['DATA_TYPE'] == 'ASCII_INTEGER':
            dtype = repeat(np.int, col)
        elif col['DATA_TYPE'] == 'CHARACTER':
            dtype = repeat('S{}'.format(col['BYTES']), col)
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
    # inf.close()

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
        for j in range(desc.get('ITEMS', 1)):
            col = desc['COLUMN'][i]
            table.columns[i].name = col['NAME']
            if 'DESCRIPTION' in col:
                table.columns[i].description = col['DESCRIPTION']

    # Save table meta data.
    for k, v in desc.items():
        if k != 'COLUMN':
            table.meta[k] = v

    return table


def read_binary_table(label, key, path='.'):
    """Read a binary table as described by the label.

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
    table : 

    Raises
    ------
    NotImpementedError
    ValueError

    """

    import numpy as np

    warn(ExperimentalFeature('Reading binary tables is not well-tested.'))

    desc = label[key]
    dtype = []
    byte = 1
    offset = np.zeros(len(desc['COLUMN']))
    scale = np.ones(len(desc['COLUMN']))
    for i, col in enumerate(desc['COLUMN']):
        if col['START_BYTE'] != byte:
            raise NotImplementedError("Table requires skipping bytes.")

        items = ''
        if col.get('ITEMS', 1) != 1:
            items = '({},)'.format(col['ITEMS'])
            warn(ExperimentalFeature(
                "Column requires multiple items per column."))

        byte += col['BYTES']
        print(i, byte)
        item_bytes = col.get('ITEM_BYTES', col['BYTES'])
        x = PDS3_DATA_TYPE_TO_DTYPE[col['DATA_TYPE']] + str(item_bytes)
        dtype.append((col['NAME'], items + x))

        offset[i] = col.get('OFFSET', 0.0)
        scale[i] = col.get('SCALE', 1.0)

    if desc['ROW_BYTES'] != byte - 1:
        raise NotImplementedError("Table requires skipping bytes.")

    dtype = np.dtype(dtype)

    if isinstance(label['^' + key], tuple):
        filename, start = label['^' + key]
        start = int(start) - 1
        filename = _find_file(filename, path=path)
        if 'RECORD_BYTES' in label:
            record_bytes = label['RECORD_BYTES']
        else:
            record_bytes = desc['RECORD_BYTES']

        inf = open(filename, 'r')
        inf.seek(record_bytes * start)
    else:
        filename = _find_file(label['^' + key], path=path)
        start = 0
        inf = open(filename, 'r')

    n = desc['ROWS']
    data = np.fromfile(inf, dtype=dtype, count=n).view(np.recarray)
    return data


def read_image(label, key, path=".", scale_and_offset=True, verbose=False):
    """Read an image as described by the label.

    The image is not reordered for display orientation.

    When there are interleaved data (i.e., the BANDS keyword is
    present), they will be separated and a data cube returned.

    Parameters
    ----------
    label : dict
      The label, as read by `read_label`.
    key : string
      The label key of the object that describes the image.
    path : string, optional
      Directory path to label/table.
    scale_and_offset : bool, optional
      Set to `True` to apply the scale and offset factors and return
      floating point data.
    verbose : bool, optional
      Print some informational info.

    Returns
    -------
    im : ndarray
      The image.  If there are multiple bands, the first index
      iterates over each band.

    """

    import os.path
    import warnings
    import numpy as np

    warnings.warn("read_image is a basic and incomplete reader.")

    # The image object description.
    desc = label[key]

    shape = np.array((desc['LINES'], desc['LINE_SAMPLES']))
    size = desc['SAMPLE_BITS'] // 8

    if 'BANDS' in desc:
        bands = desc['BANDS']
    else:
        bands = 1

    if 'LINE_PREFIX_BYTES' in desc:
        prefix_shape = (shape[0], desc['LINE_PREFIX_BYTES'])
    else:
        prefix_shape = (0, 0)

    if 'LINE_SUFFIX_BYTES' in desc:
        suffix_shape = (shape[0], desc['LINE_SUFFIX_BYTES'])
    else:
        suffix_shape = (0, 0)

    line_size = prefix_shape[0] + shape[0] * size + suffix_shape[0]

    if desc['SAMPLE_TYPE'] in PDS3_DATA_TYPE_TO_DTYPE:
        dtype = '{}{:d}'.format(PDS3_DATA_TYPE_TO_DTYPE[desc['SAMPLE_TYPE']],
                                size)
    else:
        raise NotImplemented('SAMPLE_TYPE={}'.format(desc['SAMPLE_TYPE']))

    filename, start_record = label['^{}'.format(key)]
    found_filename = _find_file(filename, path=path)
    start = (start_record - 1) * int(label['RECORD_BYTES'])

    if verbose:
        print('''Image shape: {} samples
Number of bands: {}
Stored line size, including prefix, and suffix: {} bytes
Numpy data type: {}
Filename: {} ({})
Start byte: {}'''.format(shape, line_size, bands, dtype, filename,
                         found_filename, start))

    # The file is read into a an array of bytes in order to properly
    # handle line prefix and suffix.
    with open(found_filename, 'rb') as inf:
        inf.seek(start)
        n = np.prod(line_size * shape[1])
        buf = inf.read(n)
        if n != len(buf):
            raise IOError("Expected {} bytes of data, but only read {}".format(
                n, len(buf)))

    # remove prefix and suffix, convert to image data type and shape
    data = np.frombuffer(buf, dtype=np.uint8, count=n)
    del buf
    data = data.reshape((line_size, shape[1]))
    s = slice(prefix_shape[0], (suffix_shape[0]
                                if suffix_shape[0] > 0 else None))
    data = data[s, :].flatten()
    im = np.frombuffer(data, dtype=dtype, count=np.prod(shape)).reshape(shape)
    del data

    # separate out interleaved data
    if bands > 1:
        if desc['BAND_STORAGE_TYPE'].upper() == 'SAMPLE_INTERLEAVED':
            im = im.reshape((shape[0], shape[1] // bands, bands))
            im = np.rollaxis(im, 2)
        elif desc['BAND_STORAGE_TYPE'].upper() == 'LINE_INTERLEAVED':
            im = im.reshape((shape[0] // bands, bands, shape[1]))
            im = np.rollaxis(im, 1)
        else:
            raise ValueError('Incorrect BAND_STORAGE_TYPE: {}'.format(
                desc['BAND_STORAGE_TYPE']))

    if ('OFFSET' in desc or 'SCALING_FACTOR' in desc) and scale_and_offset:
        im = (desc.get('OFFSET', 0)
              + im.astype(float) * desc.get('SCALING_FACTOR', 1))

    return im


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
        raise NotImplementedError(
            "Table format not implemented: {}".format(format))
