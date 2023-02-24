from datasets import load_dataset

datasets = load_dataset('text', data_files="/home/keonwoo/anaconda3/envs/KoDiffCSE/data/kowiki_convert.txt", cache_dir="./data/")


column_names = datasets["train"].column_names
sent2_cname = None
if len(column_names) == 2:
    # Pair datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
elif len(column_names) == 3:
    # Pair datasets with hard negatives
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
    sent2_cname = column_names[2]
elif len(column_names) == 1:
    # Unsupervised datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[0]
else:
    raise NotImplementedError




def prepare_features(examples):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the 
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    total = len(examples[sent0_cname])

    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "
    
    sentences = examples[sent0_cname] + examples[sent1_cname]

    # If hard negative exists
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=10,
        truncation=True,
        padding="max_length" if True else False,
    )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
    return features


from kobert_tokenizer import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")

train_dataset = datasets["train"].map(
    prepare_features,
    batched=True,
    num_proc=4,
    remove_columns=column_names,
    load_from_cache_file=not False,
)

print(train_dataset['train'])