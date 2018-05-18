"""
Test for comparing files - used for testing custom Parser's output
"""
def compare_files(fname, gname):
    """

    :param fname: custom parser's output
    :param gname: validated output
    :return: pass/fail
    """
    with open(fname) as f:
        fcontent = f.readlines()
    with open(gname) as g:
        gcontent = g.readlines()
    i=0
    for f in fcontent:
        if gcontent[i] != f:
            print("these don't match", f, gcontent[i])
        i=i+1
compare_files("chains.txt", "parapred-master/edgar.txt")