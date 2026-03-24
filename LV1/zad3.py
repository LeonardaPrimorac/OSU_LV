brojevi=[]

while True:
    x=input()

    if x=="Done":
        break

    try:
        broj=int(x)
        brojevi.append(broj)
    except:
        print("Pogrešan unos")

brojevi.sort()
    
    

print(f"Koliko je brojeva u listi: {len(brojevi)}")
print(f"Srednja vrijednost: {sum(brojevi)/len(brojevi)}")
print(f"Minimalna vrijednost: {min(brojevi)}")
print(f"Maksimalna vrijednost: {max(brojevi)}")
print(f"Sortirana lista {brojevi}")
