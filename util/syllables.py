from nltk.corpus import cmudict, wordnet, stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import random

d = cmudict.dict()

stop_words = stopwords.words("english")


def is_stop_word(word):
    """Returns True if the word is a stop word, False otherwise."""

    return word in stop_words


def text_prediction(text, possible_words=[]):
    """Predicts the next word in a sequence of words.

    Args:
    text: A string of text.

    Returns:
    The most likely next word in the sequence.
    """

    # Create a bag of words for the text.
    bag_of_words = FreqDist(text)

    # Create a dictionary of word probabilities.
    word_probabilities = {}
    for word, count in bag_of_words.items():
        word_probabilities[word] = count / sum(bag_of_words.values())

    # Find the word with the highest probability.
    words = sorted(word_probabilities, key=word_probabilities.get, reverse=True)
    for word in words:
        if word in possible_words:
            return word

    return random.choice(possible_words)


def nsyl(word):
    if word == "":
        return 0
    if len(word.split(" ")) > 1:
        return sum([nsyl(w) for w in word.split(" ")])
    try:
        a = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
        return min(a[0], syllable_count(word)) # might break sometimes
    except KeyError:
        # if word not found in cmudict
        return syllable_count(word)


def remove_punctuation(word):
    word = remove_parenthesis(word)
    word = word.replace("-", " ")
    word = "".join(
        c for c in word if c not in ("!", ".", ",", "?", ":", ";", '"', "'", "(", ")")
    )
    return word.strip()


def syllable_count(word):
    count = 0
    vowels = "aeiouyAEIOUY"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


# remove things in parenthesis
def remove_parenthesis(word):
    if "(" in word and ")" in word:
        word += " "
        beginning = word.split("(")[0]
        end = word.split("(")[1].split(")")[-1]
        return beginning.strip() + " " + end.strip()
    return word


def get_variants_of_word(word):
    forms = set()  # We'll store the derivational forms in a set to eliminate duplicates
    for happy_lemma in wordnet.lemmas(word):  # for each "happy" lemma in WordNet
        forms.add(happy_lemma.name())  # add the lemma itself
        for (
            related_lemma
        ) in happy_lemma.derivationally_related_forms():  # for each related lemma
            forms.add(related_lemma.name())  # add the related lemma

    """ synsets = wordnet.synsets(word)
    for synset in synsets:
        forms.add(synset.name().split(".")[0].replace("_", " ")) """

    return forms


def choose(words):
    print("Here are your options")
    for i in range(len(words)):
        print(str(i) + ": " + words[i])
    choice = input("Which one do you want? ")
    try:
        return words[int(choice)]
    except:
        print("Invalid choice")
        return choose(words)

    return random.choice(words)

def get_words(
    filename="linear_algebra.txt",
    output="output.txt",
    og={
        "1": [],
        "2": [],
        "3": [],
        # ...
    },
):
    # load similiar words and put in dictionary
    with open("./files/words/" + filename, "r") as f:
        words = set()
        dictionary = og

        for word in f.readlines():
            words.add(word.strip())
            #words.update(get_variants_of_word(word.strip()))

        print("Number of words: ", len(words))

        for idea in words:
            multiple_words = idea.split(" ").append(idea)
            if multiple_words == None: multiple_words = [idea]
            for word in multiple_words:
                syllables = nsyl(word.strip())
                if str(syllables) in dictionary:
                    if word.strip().lower() not in dictionary[str(syllables)]:
                        dictionary[str(syllables)].append(word.strip().lower())
                else:
                    dictionary[str(syllables)] = [word.strip()]

        print(dictionary)

    # save the dictionary
    with open(output, "w") as f:
        print(dictionary, file=f)

    return dictionary


# print(get_variants_of_word("integrate"))
print("Loaded syllables.py")
if __name__ == "__main__":
    # Load lyrics and count syllables
    with open("./files/lyrics/Havana.txt", "r") as f:
        pattern = []
        lyrics = f.readlines()
        for real_line in lyrics:
            syllable_pattern = []
            line = remove_punctuation(real_line)
            for word in line.split(" "):
                # print(word, syllable_count(word))
                syllable_pattern.append(nsyl(remove_punctuation(word)))
            pattern.append(syllable_pattern)
            print(real_line.strip(), f"({sum(syllable_pattern)})")

        print([sum(line) for line in pattern])

    topic_words = get_words(og={})
    commoners = get_words("common_words.txt", "common_words_output.txt")


    with open("topic_syll.txt", "w") as f:
        for syllable, words in topic_words.items():
            f.write(f"Words with {syllable} syllables: {len(words)}\n")
            for word in words:
                f.write(word + "\n")

    with open("common_words_syll.txt", "w") as f:
        for syllable, words in commoners.items():
            f.write(f"Words with {syllable} syllables: {len(words)}\n")
            for word in words:
                f.write(word + ", ")
            f.write("\n")

    pre_prompt =  """
    The New Constitution
 The Constitutional Convention: Fifty-five delegates from twelve states assembled in
Philadelphia in May 1787.
 Conflicts arose between large and small states, and free and slave states.
 The Great Compromise provided a middle ground for agreement by:
o a bicameral legislature that had one house based on population and one
representing all states equally; and
o a compromise on free-state and slave-state interests by agreeing to count five
slaves as three freemen.

 To insulate the election of the president from the popular vote, a electoral college was
created to select a president.
 Ratifying the Constitution: Supporters of the Constitution called themselves Federalists.

 Anti-Federalist opponents feared the Constitution gave too much power to the central
government and that a republic could not work well in a large nation.
 James Madison, Alexander, Hamilton, and John Jay published the influential The
Federalist that helped secure passage.
 Ratifying the New Constitution: The Bill of Rights
 Several states including Virginia, agreed to ratification only if a bill of rights would be
added.
 The first ten amendments, better known as the Bill of Rights to the Constitution served to
restrain the growth of governmental power over citizens.

Write a parody for this song
    """
    
    # make the lyrics
    with open("./files/lyrics/Havana.txt", "r") as f:
        new_lyrics = pre_prompt + ""
        lyrics = f.readlines()
        for i in range(len(lyrics)):
            line = lyrics[i]
            line = remove_punctuation(line)
            for word in line.split(" "):
                # print(word, syllable_count(word))
                syll = nsyl(remove_punctuation(word))
                choice = "".join(["_" for i in range(len(word))])
                if syll == 0:
                    continue

                if is_stop_word(word.strip().lower()) or word.strip().lower() in ["ooh", "oh", "na"]:
                    new_lyrics += word + " "
                    continue

                if syll > 1:
                    choice = random.choice(topic_words[str(syll)])
                #if syll <= 1:
                #    choice = f"({syll})"# text_prediction(new_lyrics, commoners[str(syll)])
                new_lyrics += choice + " "
            new_lyrics += "\n"
            print(line)
            print(new_lyrics.split("\n")[-1])
            print()
        print(new_lyrics)
        # print([sum(line) for line in pattern])
