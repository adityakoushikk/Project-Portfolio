import random
#48 T-cell Epitopes
a = str('YLQPRTFLL')
b = str('FIAGLIAIV')
c = str('TLDSKTQSL')
d = str('LTDEMIAQY')
e = str('RLFRKSNLK')
f = str('NLNESLIDL')
g = str('VLNDILSRL')
h = str('RLNEVAKNL')
i = str('TPINLVRDL')
j = str('ALNTLVKQL')
k = str('LITGRLQSL')
l = str('GTITSGWTF')
m = str('KEIDRLNEV')
n = str('NQKLIANQF')
o = str('RISNCVADY')
p = str('FQPTNGVGY')
q = str('TSNQVAVLY')
r = str('SPRRARSVA')
s = str('YEQYIKWPW')
t = str('VLYQDVNCT')
u = str('GSFCTQLNR')
v = str('LLQYGSFCT')
w = str('RVDFCGKGY')
x = str('SVLNDILSR')
y = str('VRFPNITNL')
z = str('VYDPLQPEL')
aa = str('FVFKNIDGY')
ab = str('GTHWFVTQR')
ac = str('LEPLVDLPI')
ad = str('LLALHRSYL')
ae = str('LPPAYTNSF')
af = str('NGVEGFNCY')
ag = str('YFPLQSYGF')
ah = str('SIIAYTMSL')
ai = str('QYIKWPWYI')
aj = str('SSTASALGK')
ak = str('VLKGVKLHY')
al = str('VTYVPAQEK')
am = str('IITTDNTFV')
an = str('LLFNKVTLA')
ao = str('RLDKVEAEV')
ap = str('RLQSLQTYV')
aq = str('GPKKSTNLV')
ar = str('RFDNPVLPF')
at = str('GLTVLPPLL')
au = str('GKYEQYIKW')
av = str('KYEQYIKWP')
aw = str('VNFNFNGLT')

#for loop to produce monte-carlo simulation (runs many times to acheive most accurate results)
for count in range (1,100000000):
    #Loop that iterates through each epitope
    for word in a.split(),:
        #random number to decide the amount of amino acids to be altered
        random_amount = int(random.randint(0, 9))
        #T-cell Resistant Sequence (Starting Sequence) in the form of a list 
        words_list = ['V', 'L', 'Y', 'Q', 'D', 'V', 'N', 'C', 'T']

        #For loop to randomly alter amino acids at a random position in the T-cell resistant sequence
        for count in range (0,random_amount):
            randomPlacement = random.randint(0,8)
            integer = int(randomPlacement)
            words_list[integer] = random.choice(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])



        #After randomly altering amino acid, compile the list of newly mutated amino acids into a single sequence as a string
        new_word = str(words_list[0] + words_list[1] + words_list[3] + words_list[4] + words_list[5] + words_list[6] + words_list[7] + words_list[8])

    #Conditionals to check if the new word matches any of the T-cell epitopes. If a match is found, display the number of different cycles (loops) it took to do so
    if (new_word == a or new_word == b or new_word == c or new_word == d or new_word == e or new_word == f or new_word == g or new_word == h or new_word == i or new_word == j or new_word == k or
        new_word == l or new_word == m or new_word == n or new_word == o or new_word == p or new_word == q or new_word == s or new_word == t or new_word == u or new_word == v or new_word == w or
        new_word == x or new_word == y or new_word == z or new_word == aa or new_word == ab or new_word == ac or new_word == ad or new_word == ae or new_word == af or new_word == ag or new_word == ah
        or new_word == ai or new_word == aj or new_word == ak or new_word == al or new_word == am or new_word == an or new_word == ao or new_word == ap or new_word == aq or new_word == ar or new_word == at or new_word == au
        or new_word == av or new_word == aw):
        print(count)
        print(new_word)
        break
