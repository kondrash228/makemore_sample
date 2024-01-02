from pprint import pprint
import torch
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()
# this is a bigram model which is work only with 2 chars in a time and try to predict the next char

# then we create a two dimensional array where rows are going to be the first character and the columns are going to be the second character
# and it'll show us how often the second character follows the first
N = torch.zeros((27,27), dtype=torch.int32) # N[1,3] = 1 indexing array in pytorch

chars = sorted(list(set(''.join(words)))) # just getting all characters which we have in dataset

stoi = {s:i + 1 for i, s in enumerate(chars)} # mapping character to the integer
stoi["."] = 0 # special character for determening start and end of word

itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ["."] + list(w) + ["."] # we're gonna to count how many times this combinations of characters are occurs in the training dataset
 
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        
        N[ix1, ix2] += 1 # counting how many times this sequense of characters occurs in dataset 



# visualize our two-dimensional array
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')

for i in range(27):
    for j in range(27):
        chsrt = itos[i] + itos[j]
        plt.text(j,i,chsrt, ha="center", va="bottom", color="gray")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.show()


# roughly speaking we gonna use the probabilities from this model (two-dimensional array) to start sampling from the model
# we have to create probabilites from these counts

"""
we're expecting:
tensor([0.6064, 0.3033, 0.0903])

0 (60% prob)
1 (30% prob)
2 (10% prob)

and we get 20 samples with respective data
tensor([1, 1, 2, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]) 
most of nums are 0
less 1
and very few is 2
"""

# create a matrix with probs which are normalized

P = N.float()
P = P / P.sum(1, keepdim=True) # we can divide like this because of broadcast semantics in torch

g = torch.Generator().manual_seed(2147483647)

for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # p = N[ix].float()
        # p = p / p.sum() # create probabilty distribution by dividing it by p.sum in another words we normalize it

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item() # helps us to choose a char based on it's probability from two-dimensional array
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))


# ends in 47:37

