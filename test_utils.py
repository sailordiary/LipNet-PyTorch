import sys


def count_net_params(net, show_details=False):
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        if show_details: print ("Struct:", list(i.size()), " -- %d params" % l)
        k += l
    if show_details: print ("%d params in total" % k)
    return k


def test_network(net, show_count=True):
    print (net)
    if show_count: print (count_net_params(net, True))
    sys.exit()
