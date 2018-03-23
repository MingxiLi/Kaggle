import pandas as pd
X = []
X.append([1, 2])
X.append([3, 4])
print(X[1])
y = ['1', '2']
trn = pd.DataFrame(X)
trn['label'] = y
