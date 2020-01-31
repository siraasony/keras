from numpy import array
                # 10, 4
def split_sequence(sequence, n_steps): #n_steps: 몇개씩 자를건가
    x, y = list(), list()
    for i in range(len(sequence)):       # 10
        end_ix = i + n_steps             # 0 + 4 = 4
        if end_ix > len(sequence)-1 :    # 4 > 10 -1
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix] #  seq_x = 0,1,2,3 / seq_y = 4
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

dataset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

n_steps = 3
X, y = split_sequence(dataset, n_steps)

print (X)
print (y)

'''
0, 1, 2, 3 / 4
1, 2, 3, 4 / 5

5, 6, 7, 8 / 9
'''