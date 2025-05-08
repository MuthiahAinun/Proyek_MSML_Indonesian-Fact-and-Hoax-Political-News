from augment import eda  # import dari augment.py

def load_and_augment_dataset(path, do_augment=True):
    df = pd.read_csv(path, compression='gzip')
    df['label'] = df['label'].astype(int)

    if do_augment:
        hoax_df = df[df['label'] == 1]
        augmented_texts = []
        for text in hoax_df['text']:
            augmented_texts.extend(eda(text, num_aug=2))

        augmented_df = pd.DataFrame({'text': augmented_texts, 'label': 1})
        df = pd.concat([df, augmented_df], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)

    return df
