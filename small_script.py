import numpy as np
import boardchange_km as bc
import training_input_cole as inp
import tarfile
import positions_evaluation as ev

#tar = tarfile.open("amateur4d.tar.gz", 'r:gz')
res = 0
#fout = open('filenames_score.txt', 'w')
with open('filenames_score.txt', 'r') as filenames:
    for num, line in enumerate(filenames):
    #        if num < 5000:
    #            continue
    #        print(line)
        f = open('/Users/user/Downloads/amateur4d/' + line[:-1], 'r')
        data = f.read()
        print(data.count(';'))
#        print(batch_out.shape)
#        print(batch_out[20])
print('Cool!')