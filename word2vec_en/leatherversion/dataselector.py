import glob


def dataselect():
    contents = ['bags', 'shoes', 'wallet']
    adverbs = ['recent',
               'fresh',
               'advanced',
               'brand-new',
               'current',
               'different',
               'late',
               'modern',
               'original',
               'state-of-the-art',
               'strange',
               'unfamiliar',
               'unique',
                #    'unusual'
                #    'aged',
               'ancient',
               'decrepit',
               'elderly',
               'gray',
               'mature',
               'tired',
               'venerble',
               'fossil',
               'senior',
               'versed',
               'veteran',
               'broken',
               'debilitated',
               'enfeebled',
               'exhausted',
               ]
    dataset = []
    print('making word and path sets......')
    for cont in contents:
        for adv in adverbs:
            word = adv
            if adv == 'brand-new':
                word = 'new'
            elif adv == 'state-of-the-art':
                word = 'fashionable'
            elif adv == 'venerble':
                word = 'venerable'
            dir = '/home/yanai-lab/sugiya-y/space/crawl/leather_img/' + cont + '_' + adv + '/*'
            for path in glob.glob(dir):
                dataset.append([word, path])
    return(dataset)


if __name__ == '__main__':
    dataloader()
