"""
Outputs descriptions of ILSVRC synsets, ordered to work with Google's
pre-trained models.
"""
descriptions = {}
with open('imagenet_metadata.txt', 'rb') as f:
    for line in f:
        synset, description = line.split(b'\t')
        descriptions[synset.strip()] = description
outputs = [b'background\n']
with open('imagenet_lsvrc_2015_synsets.txt', 'rb') as f:
    for line in f:
        outputs.append(descriptions[line.strip()])
with open('descriptions.txt', 'wb') as f:
    f.writelines(outputs)
