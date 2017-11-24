def compare_files(fname, gname):
    with open(fname) as f:
        fcontent = f.readlines()
    with open(gname) as g:
        gcontent = g.readlines()
    i=0
    for f in fcontent:
        if gcontent[i] != f:
            print("these don't match", f, gcontent[i])
        i=i+1
compare_files("chains.txt", "edgar-parapred/edgar.txt")