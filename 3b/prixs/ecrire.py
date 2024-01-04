#! /usr/bin/python3

ema = lambda l,K,ema=0: [ema:=(ema*(1-1/K) + e/K) for e in l]

from sys import argv

prixs         = argv[1]
sortie_prixs  = "prixs.bin"
sortie_volume = "volumes.bin"
sortie_macd   = "macds.bin"

import struct as st

#	======== Lecture .csv ======= 
with open(prixs, "r") as co:
	text = co.read().split('\n')
	del text[0]
	del text[0]
	del text[-1]
	lignes = [l.split(',') for l in text][::-1] # <-- Important le [::-1]
	infos = [(float(Open), float(Volume_BTC), float(Volume_USDT)) for Unix,Date,Symbol,Open,High,Low,Close,Volume_BTC,Volume_USDT,tradecount in lignes]

#	========= Ecriture ==========
prixs   = [p       for p,_,_   in infos]
s=0; volumes = [s:=(s + (vb*p-vu)) for p,vb,vu in infos]
ema12 = ema(prixs, K=12)
ema26 = ema(prixs, K=26)
macd  = [a-b for a,b in zip(ema12, ema26)]
ema9_macd = ema(macd, K=9)
histo_macd = [a-b for a,b in zip(macd, ema9_macd)]

with open(sortie_prixs, "wb") as co:
	print(f"LEN prixs   = {len(prixs)}")
	co.write(st.pack('I', len(prixs)))
	co.write(st.pack('f'*len(prixs), *prixs))

with open(sortie_volume, "wb") as co:
	print(f"LEN volumes = {len(volumes)}")
	co.write(st.pack('I', len(volumes)))
	co.write(st.pack('f'*len(volumes), *volumes))

with open(sortie_macd, "wb") as co:
	print(f"LEN macd    = {len(histo_macd)}")
	co.write(st.pack('I', len(histo_macd)))
	co.write(st.pack('f'*len(histo_macd), *histo_macd))