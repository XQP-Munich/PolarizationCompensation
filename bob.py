import numpy as np

def read_file(filename):
    with open(filename,"r") as file:
        cps = [[],[],[],[]]
        for line in file.readlines()[1:]:
            x,y1,y2,y3,y4,y5 = line.split("\t")
            y1=int(y1)
            y2=int(y2)
            y3=int(y3)
            y4=int(y4)
            if y1 ==0:
                continue
            cps[0].append(y1)
            cps[1].append(y2)
            cps[2].append(y3)
            cps[3].append(y4)
    return np.array(cps).mean(axis=1)


single_apd = read_file("./data/bob_characterization/Counter_2024-08-06_16-14-35.txt")[0]/10
bob = read_file("./data/bob_characterization/Counter_2024-08-06_16-31-09.txt")*10

eff_bob_without_bp = bob.sum()/single_apd


print("Bob loss without BP: {:.2f}% in db: {:.2f}".format(eff_bob_without_bp*100,np.log10(eff_bob_without_bp)*10))


counts_nd_bp_4ch= read_file("./data/bob_characterization/Counter_2024-08-06_16-56-32.txt").sum()*10
counts_nd_4ch =  read_file("./data/bob_characterization/Counter_2024-08-06_16-53-21.txt").sum()*10


eff_bp = counts_nd_bp_4ch/counts_nd_4ch

print("Bandpass loss: {:.2f}% in db: {:.2f}".format(eff_bp*100,np.log10(eff_bp)*10))

eff_apd = 0.432169
eff_bob_total = eff_bp*eff_bob_without_bp*eff_apd

print("Bob total loss with apd and bp: {:.4f}% in db: {:.2f}".format(eff_bob_total*100,np.log10(eff_bob_total)*10))

eff_nd = 0.04528

counts_alice_set1 = (904963+875032+784966+838220)/10 /eff_bob_total/((1.320/100))
rep_rate = 100E6

print("Mean Photon Nr.:{:.2f}".format(counts_alice_set1/rep_rate))




a=np.array([1641.7263951718585, 3513.719708029197, 1563.2760559033018, 1717.2146526983831])

b=np.array([1585.2858845032774, 4371.281657265124, 4574.453697332266, 5255.506438956009])

c=a-b
print(c)

[1589.982070134927, 3516.0989681050655, 1586.9446713892876, 1597.0507746055382]
[1581.0280105128925, 3527.270658682635, 1579.2819311364185, 1589.115262219051]