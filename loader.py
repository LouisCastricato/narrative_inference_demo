import json

data = None
with open("mem_beam_outputs.jsonl") as f:
    data = json.load(f)

inp = data["<|sent0|>_generated_relations"]

xneed = inp[0]
xintent = inp[1]
xwant = inp[2]
oeffect = inp[3]
xreact = inp[4]
owant = inp[5]
oreact = inp[6]
xeffect = inp[7]
xattr = inp[8]

def clean(s):
    spl = s.split()
    for i, word in enumerate(spl):
        if word == "personx":
            spl[i] = "PersonX"
        if word == "persony":
            spl[i] = "PersonY"
        if word == "personx's":
            spl[i] = "PersonX's"
        if word == "persony's":
            spl[i] = "PersonY's"
    return " ".join(spl)
    
string = ""


for s in owant:
    string += "<oEffect> " + clean(s) + " "

for s in xintent:
   string += "<xEffect> " + clean(s) + " "

for s in xneed:
    string += "<xWant> " + clean(s) + " "

#for s in xattr:
#    string += "<xReact> " + s + " "
print(string)
