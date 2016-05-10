import numpy as np
import boardchange_km as bc
import training_input_cole as inp
import tarfile
import positions_evaluation as ev

tar = tarfile.open("amateur_batch2.tar.gz", 'r:gz')
with open('filenames_kgs_batch.txt', 'r') as filenames:
    for num, line in enumerate(filenames):
#        if num < 5000:
#            continue
#        print(line)
        member = tar.getmember("./amateur_batch2/" + line[:-1])
        f = tar.extractfile(member)
        if num % 100 == 0:
            print(num)
#        print(batch_out.shape)
#        print(batch_out[20])
print('Cool!')