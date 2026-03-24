ham_broj_rijeci = 0
spam_broj_rijeci = 0

ham_broj_poruka = 0
spam_broj_poruka = 0

spam_usklicnik = 0

with open("SMSSpamCollection.txt", encoding="utf-8") as datoteka:
    for linija in datoteka:
        linija = linija.strip()

        dijelovi = linija.split(None, 1)  

        if len(dijelovi) != 2:
            continue

        tip, poruka = dijelovi

        broj_rijeci = len(poruka.split())

        if tip == "ham":
            ham_broj_poruka += 1
            ham_broj_rijeci += broj_rijeci

        elif tip == "spam":
            spam_broj_poruka += 1
            spam_broj_rijeci += broj_rijeci

            if poruka.endswith("!"):
                spam_usklicnik += 1

prosjek_ham = ham_broj_rijeci / ham_broj_poruka if ham_broj_poruka else 0
prosjek_spam = spam_broj_rijeci / spam_broj_poruka if spam_broj_poruka else 0

print("Prosječan broj rijeci (ham):", prosjek_ham)
print("Prosječan broj rijeci (spam):", prosjek_spam)
print("Broj spam poruka koje završavaju usklicnikom:", spam_usklicnik)