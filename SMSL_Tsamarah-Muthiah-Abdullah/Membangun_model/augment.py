import random
import re

def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    return [word for word in words if random.uniform(0, 1) > p]

def random_swap(words, n=1):
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def eda(text, num_aug=4):
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    augmented_sentences = []

    for _ in range(num_aug):
        aug_type = random.choice(['swap', 'delete'])
        if aug_type == 'swap':
            new_words = random_swap(words, n=max(1, len(words) // 10))
        else:
            new_words = random_deletion(words, p=0.1)
        augmented_sentences.append(' '.join(new_words))

    return augmented_sentences
