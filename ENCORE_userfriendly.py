# coding:gbk
'''
Created on 2019-8-15

@author: Bingbo Wang and Annika Rohl
'''

# INPUT:
# DEfilename: 	Includes the experimental data. Three columns, seperated by tab (\t). column 1) gene name, string
#               (like in the network), 2) p-value (float), 3) log fold change (float). If row starts with '#':
#               description, row will be ignored
# PPIfilename:  Underlying interactome. Two columns, seperated by tab (\t). Each row gives an aedge (column 1) <-> (column 2))

import networkx as nx
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

G2 = nx.Graph()


def Pattern_detection(DEfilename, PPIfilename):
    global G2  # the PPI as a networkx graph
    openPPI(PPIfilename)  # read in the network given in the PPIfilename

    genenumber, fc = readdata(DEfilename)  # list of significantly differentially express genes in the PPI (genenumber) and their corresponding fold change values (fc)

    zlist, fclist, size = lcc_zscore(genenumber, fc, 100, 50, max(fc))  # zlist: z-scores, fclist: list of fold change cutoffs, size: list of sizes of the LCCs
    max_fcvalue = refine(zlist, fclist, size)  # we redefine the max fold change, depending on the sizes of the LCCs, exclude the small LCCs

    zlist, fclist, size = lcc_zscore(genenumber, fc, 1000, 50, max_fcvalue)  # do as before with updated max fold change value

    p1, fccutoff1, fccutoff2, minpick, sig = leastsqfit(fclist, zlist)  # compute polynomial, the two fold change cut offs for the two peaks,

    print('FC cutoffs for disease neighborhood and core:', pow(2, fccutoff1), pow(2, fccutoff2))

    DrawPicture(DEfilename, pow(2, max_fcvalue), max(zlist) + 2, p1, zlist, fclist, fc)  # draw fitting curve, zscores, and fold changes

    module, coremodule = getlist(fccutoff1, fccutoff2, genenumber,fc)  # get lists of genes in the core and disease neighborhood

    core = getLCC(G2, coremodule)  # compute LCC given by the core genes
    Dmodule = getLCC(G2, module)  # compute LCC given by the disease neighborhood genes
    periphery = list(set(Dmodule) - set(core))  # peripheral genes are the genes in the disease neighborhood excluding the core genes
    g = nx.subgraph(G2, Dmodule)
    gg = nx.subgraph(G2, core)
    file_name_neighborhood = 'disease_neighborhood.txt'  # file for the disease neighborhood
    if os.path.exists(file_name_neighborhood):  # check if file already exists
        os.remove(file_name_neighborhood)
    f = open(file_name_neighborhood, 'a')  # create file
    for edge in g.edges:  # loop over edges
        f.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')  # write down an edge per row
    f.close()  # close file
    file_name_core = 'core_and_periphery_genes.txt'
    if os.path.exists(file_name_core):
        os.remove(file_name_core)
    f = open(file_name_core, 'a')
    f.write('#GeneSymbol\tCategories\n')
    for x in core:
        f.write(str(x) + '\tcore\n')
    for x in periphery:
        f.write(str(x) + '\tperiphery\n')
    f.close()
    return g, gg


def openPPI(filename):
    global G2
    file_network = open(filename, 'rb')
    G2 = nx.read_weighted_edgelist(file_network)  # read in graph from file
    file_network.close()


def readdata(filename):
    global G2  # PPI
    p = 0.05  # p-value, only significantly differentially expressed genes are considered
    genename = []  # initialise list of names of significant genes
    fc = []  # initialise list of fold change value of the genes
    pvalue = []  # initialise list of pvalues of the genes
    first = open(filename, "r")  # open the file which contains the names, pvalues, and fold change of the genes
    for line in first:  # iterate through the file
        i = line.strip()  # prepare
        if not i.startswith('#'):  # check if comment
            n = i.split("\t")  # symbols, pvalues and log fold changes are stored in columns, seperated by tab
            if (float(n[1]) <= p):  # only significantl differentially expressed genes, thus a p-value smaller than 0.05
                pvalue.append(float(n[1]))  # p-value of the gene is added to the list
                if n[0] in G2:  # check if gene is in the PPI
                    fc.append(abs(float(n[2])))  # log fold change value of the gene is added to the list
                    genename.append(n[0])  # gene name is added to the list

    print('Number of DE genes with p-value<0.05:', len(fc), 'Found in PPI:', len(genename))
    return genename, fc


def lcc_zscore(genelist, fclist, ran, freq, maxfc):
    import random
    global G2  # The PPI
    if ran <= 100:
        print("Preprocessing...\n")

    zscore = []  # zscores of the LCCs
    fccutoff = []  # foldchange cut offs related to the LCCs
    sizeoflcc = []  # sizes of the LCCs
    alfa = (maxfc - min(fclist)) / freq  # fold change cutoff increment

    # loop, where in each step we compute a LCC given by the new fold change cut off
    for u in range(freq):
        use_genes = []  # genes which we use to compute the subgraph
        cutoff = min(fclist) + alfa * (u + 1)  # fold change cut off
        for x in range(len(genelist)):
            if (fclist[x] >= cutoff):  # collect all genes which have a fold change higher than the cut off
                use_genes.append(genelist[x])

        g = nx.subgraph(G2, use_genes)  # subgraph given by all genes of fold change higher than the cutoff
        largest_cc = max(nx.connected_components(g),
                         key=len)  # LCC given by all genes of fold change higher than the cutoff

        largest = len(largest_cc)  # size of largest connected component

        if ran > 100:
            print('size of LCC:', largest)
        sizeoflcc.append(largest)

        number_of_sims = ran  # number of simulations
        all_genes = G2.nodes()  # all the genes in th PPI
        l_list = []

        # simulations with randomly distributed seed nodes
        for i in range(number_of_sims):
            black_nodes = random.sample(all_genes, len(
                use_genes))  # randomly selected nodes, same number as all genes of fold change higher than the cutoff

            g2 = nx.subgraph(G2, black_nodes)  # subgraph given by the randomly selected genes
            LCC2 = max(nx.connected_components(g2), key=len)  # LCC given by the randomly selected genes
            largest2 = len(LCC2)  # size of the LCC given by the randomly selected genes

            l_list.append(largest2)  # list of sizes of LCC

        l_mean = np.mean(l_list)  # mean  of sizes of LCC
        l_std = np.std(l_list)  # standard deviation of sizes of LCC

        if l_std == 0:
            z_score = 0
        else:
            z_score = (1. * largest - l_mean) / l_std  # compute z-score

        zscore.append(z_score)  # update list of z-scores
        fccutoff.append(cutoff)  # update list of fold change cut off values
    return (zscore, fccutoff, sizeoflcc)


def refine(zlist, fclist, size):
    flag = 0
    max_fcvalue = max(fclist)
    for i in range(len(zlist)):
        if size[i] > 5:
            flag = 0
        else:
            flag = flag + 1  # if LCC is too small
        if flag >= 3:
            max_fcvalue = fclist[i]
            break
    return max_fcvalue


def leastsqfit(fclist, zlist):
    from scipy.optimize import leastsq
    z = []
    for i in range(len(zlist)):
        z.append(pow(zlist[i], 3))  # zscore^3

    z1 = np.polyfit(fclist, z, 8)  # least square poly fitting of the z-scores to the power of 3 to the fold change values using a polynomial of degree 8

    p1 = np.poly1d(z1)  # polynomial
    z2 = np.polyfit(fclist, zlist, 8)  # least square poly fitting of the original z-scores to the fold change values using a polynomial of degree 8
    p2 = np.poly1d(z2)  # polynomial

    f = []  # initialise list for the values given by the polynomial at the fold change values
    maximum = []  # initialise list with maximum values of the fitting curve
    maximumfclist = []  # initialise list with fold change values related to the peaks of the fitting curve
    for i in range(len(fclist)):
        f.append(p1(fclist[i]))  # evaluate the polynomial p1 at the foldchange values.

    # find maximum peaks in the curve:
    if f[0] > f[1]:
        maximum.append(f[0])
        maximumfclist.append(fclist[0])
    for i in range(1, len(fclist) - 1):
        if f[i] > f[i - 1] and f[i] > f[i + 1]:
            maximum.append(f[i])
            maximumfclist.append(fclist[i])

    s = maximum.index(max(maximum))  # index for max peak

    fccutoff1 = maximumfclist[s]  # fold change value for the max peak
    del maximum[s]  # remove so next max can be found
    del maximumfclist[s]  # remove so next max can be found
    if len(maximum) < 1:  # stop criteria
        fccutoff2 = fccutoff1
    else:
        s = maximum.index(max(maximum))
        fccutoff2 = maximumfclist[s]

    if fccutoff1 > fccutoff2:
        t = fccutoff1
        fccutoff1 = fccutoff2
        fccutoff2 = t

    # find first peak's fold change value
    p = []  # initialise list for zscores
    s = fclist.index(fccutoff1)  # index of the first cut off (fold change value)
    r = s + 5
    if r >= 50:
        r = 50
    if s >= 5:
        for i in range(s - 5, r):
            p.append(zlist[i])
    if s < 5:
        for i in range(10):
            p.append(zlist[i])
    s = zlist.index(max(p))
    fccutoff1 = fclist[s]

    # find second peak's fold change value
    p = []
    s = fclist.index(fccutoff2)
    r = s + 5
    if r >= 50:
        r = 50
    if s >= 5:
        for i in range(s - 5, r):
            p.append(zlist[i])
    if s < 5:
        for i in range(10):
            p.append(zlist[i])

    s = zlist.index(max(p))
    fccutoff2 = fclist[s]

    # find minimum zscores
    minfc = []
    minzscore = []
    for i in range(len(fclist)):
        if (fccutoff1 <= fclist[i]) & (fclist[i] <= fccutoff2):
            minfc.append(fclist[i])
            minzscore.append(zlist[i])

    e = minzscore.index(min(minzscore))
    minpick = minfc[e]  # fold change of the mimium z-score of the vally between the two peaks
    sig = abs(p2(fccutoff2) - p2(minpick)) / p2(fccutoff1)  # dratio

    return p2, fccutoff1, fccutoff2, minpick, sig


def DrawPicture(titlename, xlim, ylim, p1, zlist, fclist, fc):
    # titlename: DEfilename, xlim: pow(2,max_fcvalue), ylim: max(zlist)+2, p1, zlist, fclist: fold change cut offs, fc: fold changes of the original data set)

    size = 11
    matplotlib.rc('xtick', labelsize=size - 1)
    matplotlib.rc('ytick', labelsize=size - 1)

    newfc = []
    newfclist = []
    for i in range(len(fc)):
        newfc.append(pow(2, abs(fc[i])))  # absolute values of the fold changes to the power of 2
    for i in range(len(fclist)):
        newfclist.append(pow(2, fclist[i]))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=600)

    axes.plot(newfclist, zlist, 'blue')
    axes.plot(newfclist, zlist, 'ro')

    results, edges = np.histogram(newfc, bins=100, normed=True)
    binWidth = edges[1] - edges[0]
    axes.bar(edges[:-1], 16 * results * binWidth, binWidth, color='seagreen')

    axes.plot(newfclist, p1(fclist), 'gold')
    axes.set_title(titlename, fontsize=size)

    # adding horizontal grid lines

    axes.set_xlabel('Fold Change Cutoff', fontsize=size)
    axes.set_ylabel('Z-score', fontsize=size)
    axes.set_ylim([-2, ylim])
    axes.set_xlim([1, xlim])

    plt.show()
    fig.savefig(titlename+'_ENCOREcurve.svg',dpi=600, bbox_inches='tight')


def getLCC(G, genes):
    g = nx.subgraph(G, genes)  # subgraph given by the randomly selected genes
    LCC = max(nx.connected_components(g), key=len)  # LCC given by the randomly selected genes
    return LCC


def getlist(cutoff1, cutoff2, genenumber, fc):
    module = []  # initialise list of disease neighborhood genes
    coremodule = []  # initialise list of core genes

    for x in range(len(genenumber)):
        if (fc[x] >= cutoff1):
            module.append(genenumber[x])
        if (fc[x] >= cutoff2):
            coremodule.append(genenumber[x])
    return module, coremodule