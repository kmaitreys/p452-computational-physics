# import os
import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.io as io
from astropy.table import Column
import pylab as pl
pl.ion()

names = ['freq', 'freq_err', 'logI', 'df', 'El_cm',
         'gu', 'tag', 'qncode', 'qn', 'specname']


def load_spec(tag):
    """"""
    tb = io.ascii.read(
        'specdata/{}.txt'.format(tag),
        col_starts=[0, 14, 21, 29, 31, 41, 44, 51, 55, 81],
        col_ends=[13, 20, 28, 30, 40, 43, 50, 54, 80, 100],
        format='fixed_width_no_header',
        names=names)
    Eu_cm = Column(name='Eu_cm',
                   data=tb['El_cm']*(1/u.cm)+(tb['freq']*u.MHz/c.c).to(1/u.cm))
    tb.add_column(Eu_cm)
    return tb


def compute_qpart(table, T):
    """"""
    q = 1
    for row in table:
        q += row['gu']*np.exp(-row['Eu_cm']*(1/u.cm)*c.h*c.c/(c.k_B*T*u.K))
    return np.log10(q)


def get_qpart(tag):
    """"""
    if str(tag)[-3:-2] == '5':
        db = 'cdms'
        col_starts = [0, 7, 31, 45]+[45+13*(i+1) for i in range(10)]
        col_ends = [i-1 for i in col_starts[1:]] + [1000]
        names = ['tag', 'molecule', '#lines', '1000', '500', '300',
                 '225', '150', '75', '37.5', '18.75', '9.375', '5.000',
                 '2.725']
        tb_qpart = io.ascii.read(
            'partfunc/{}.part'.format(db),
            format='fixed_width_no_header',
            guess=False,
            col_starts=col_starts,
            col_ends=col_ends,
            data_start=2,
            comment='=',
            delimiter=' ',
            fill_values=('---'),
            names=names)
    elif str(tag)[-3:-2] == '0':
        db = 'jpl'
    else:
        print "Check tag value"
    return tb_qpart[tb_qpart['tag'] == tag]

def plot():
    """"""
    temps = np.linspace(0,1000,10000)
    qval = compute_qpart(tb, temps)
    tb_qpart = get_qpart(tag)
    pl.figure(1)
    pl.clf()
    pl.plot(temps,qval)
    for key in tb_qpart.keys()[3:]:
        pl.scatter(key, tb_qpart[key])
    pl.xscale('linear')
    pl.xlim([0,20])
    return qval

if __name__ == '__main__':
    tag = 28503
    T = 9.375
    tb = load_spec(tag)
    # tb.pprint()
    qval = compute_qpart(tb, T)
    tb_qpart = get_qpart(tag)
    print plot()
    
