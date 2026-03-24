try:
    x=float(input("Unesite broj između 0 i 1: "))
    if x>=1 or x<=0:
        print("Unesite broj manji od 1 ili veći od 0")
    elif x>=0.9 and x<1:
        print("A")
    elif x>=0.8 and x<0.9:
        print("B")
    elif x>=0.7 and x<0.8:
        print("C")
    elif x>=0.6 and x<0.7:
        print("D")
    else:
        print("F")
except:
    print("Nema unosa.")
