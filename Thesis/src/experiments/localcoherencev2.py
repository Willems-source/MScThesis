import random
import math
# If you already have local_coherence.py, just import from there.
# Otherwise, paste your earlier definitions (build_entity_grid, compute_coherence_probability, etc.) here.

import spacy
nlp = spacy.load("en_core_web_sm")

############################################
# 1) RE-DEFINE THE CONSTANTS AND HELPERS
############################################
class Constants:
    NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS", "PRP"}
    
    SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}
    OBJECT_DEPS  = {"dobj", "iobj", "obj", "pobj"}

    SUB    = "S"
    OBJ    = "O"
    OTHER  = "X"
    NOSHOW = "-"

def get_role(dep: str):
    if dep in Constants.SUBJECT_DEPS:
        return Constants.SUB
    elif dep in Constants.OBJECT_DEPS:
        return Constants.OBJ
    else:
        return Constants.OTHER

import pandas as pd
from collections import defaultdict

def build_entity_grid(text: str):
    doc = nlp(text)
    sentences = list(doc.sents)
    num_sents = len(sentences)

    entity_roles = defaultdict(lambda: [Constants.NOSHOW] * num_sents)

    for sent_i, sent in enumerate(sentences):
        for token in sent:
            # If it's a noun/pronoun by PTB tag
            if token.tag_ in Constants.NOUN_TAGS:
                lemma = token.lemma_.lower()
                role = get_role(token.dep_)
                entity_roles[lemma][sent_i] = role

    grid_df = pd.DataFrame(entity_roles)
    return grid_df

def _get_column_probability(column_series):
    """
    Compute average log-prob of transitions for a single entity (column).
    Closer to 0.0 (less negative) => higher coherence for that entity's chain.
    """
    column = list(column_series)
    total_sents = len(column)
    if total_sents < 2:
        return 0.0  # With just 1 sentence, can't do much.

    # Count how often each role appears
    role_counts = {}
    for r in column:
        role_counts[r] = role_counts.get(r, 0) + 1

    # Count transitions
    transitions = {}
    for i in range(1, total_sents):
        prev_r = column[i - 1]
        curr_r = column[i]
        transitions[(curr_r, prev_r)] = transitions.get((curr_r, prev_r), 0) + 1

    # Build the sequence of log-probs
    log_probs = []
    # Probability of the first role
    first_role_prob = (role_counts[column[0]] / total_sents)
    log_probs.append(math.log(first_role_prob + 1e-9))

    # Probability of subsequent transitions
    for i in range(1, total_sents):
        prev_r = column[i - 1]
        curr_r = column[i]
        transitions_count = transitions.get((curr_r, prev_r), 0)
        prev_count = role_counts.get(prev_r, 0)
        cond_prob = (transitions_count / prev_count) if prev_count > 0 else 1e-9
        log_probs.append(math.log(cond_prob + 1e-9))

    avg_log_prob = sum(log_probs) / len(log_probs)
    return avg_log_prob

def compute_coherence_probability(text: str):
    """
    Return the average (over entities) of average log-prob of role transitions.
    Typically negative. Less negative => higher local coherence.
    """
    grid = build_entity_grid(text)
    if grid.empty:
        return float("-inf")
    scores = []
    for col in grid.columns:
        col_prob = _get_column_probability(grid[col])
        scores.append(col_prob)
    return sum(scores) / len(scores)

############################################
# 2) DEMO: ORIGINAL VS. SCRAMBLED TEXT
############################################

def scramble_sentences(text):
    # Split text into a list of sentences (using spaCy or a naive approach).
    # For simplicity, let's do a naive approach by splitting on '.' or '\n'.
    # But for a cleaner approach, re-parse with spaCy.
    doc = nlp(text)
    sents = [sent.text.strip() for sent in doc.sents]
    scrambled = random.sample(sents, len(sents))
    return " ".join(scrambled)

# EXAMPLE: 6 sentences that are "locally coherent" but arguably "thematically incoherent"
# original_text = """
# Alice visited Paris to see the Eiffel Tower.
# Bob read a complex novel about astrophysics.
# Alice shopped for souvenirs in the markets.
# Bob cooked a traditional Italian meal for dinner.
# Alice admired the art at the Louvre.
# Bob went jogging in the park at sunrise.
# """

# original_text = """
# President Biden has met with various international leaders on multiple occasions, discussing economic policies, security concerns, and environmental initiatives repeatedly in conferences and meetings, all while President Biden emphasized the importance of diplomacy and cooperation in global affairs. 
# LeBron James, known for his skills on the basketball court, has not only led his team through numerous victories but LeBron James has also been a vocal advocate for social justice, community programs, and educational reform, with LeBron James often speaking at events and leading initiatives. 
# President Biden outlined strategies for improving healthcare and infrastructure in rural areas, reaffirming that President Biden will work tirelessly with Congress and community leaders to bring positive change, just as President Biden has promised during multiple press conferences. 
# LeBron James has visited several schools, interacted with fans, and invested in young athletes’ futures, with LeBron James showing his commitment to youth and community outreach again and again. 
# Despite facing criticism, President Biden maintained that President Biden’s policies would benefit the middle class, and that President Biden’s administration is committed to transparency and progress, reaffirming these points in interviews and public statements. 
# Meanwhile, LeBron James continued to dominate on the court and off the court, with LeBron James mentoring younger players, organizing charity events, and speaking out about important social issues, demonstrating that LeBron James is not just an athlete but a leader in his community.
# """
original_text = """President Biden addressed climate change, emphasizing renewable energy and environmental protection while advocating stronger international cooperation. LeBron James showcased his skills during a high-stakes basketball game, demonstrating strategic plays and leadership on the court. President Biden unveiled a healthcare reform plan, highlighting benefits for middle-class families and stressing the need for bipartisan support. LeBron James engaged in community outreach, visiting schools and organizing youth programs to promote education and social responsibility. President Biden discussed economic growth strategies, citing job creation and infrastructure improvements as key drivers of future prosperity. LeBron James participated in philanthropic events, donating to charities and supporting underprivileged communities."""
#original_text = """South Korea registered a trade deficit of $101 million in October, reflecting the country's economic sluggishness, according to government figures released Wednesday. Preliminary tallies by the Trade and Industry Ministry showed another trade deficit in October, the fifth monthly setback this year, casting a cloud on South Korea's export-oriented economy. Exports in October stood at $5.29 billion, a mere 0.7% increase from a year earlier, while imports increased sharply to $5.39 billion, up 20% from last October. South Korea's economic boom, which began in 1986, stopped this year because of prolonged labor disputes, trade conflicts and sluggish exports. Government officials said exports at the end of the year would remain under a government target of $68 billion. Despite the gloomy forecast, South Korea has recorded a trade surplus of $71 million so far this year. From January to October, the nation's accumulated exports increased 4% from the same period last year to $50.45 billion. Imports were at $50.38 billion, up 19%."""

if __name__ == "__main__":
    # 1) Compute local coherence of the original text
    original_score = compute_coherence_probability(original_text)
    print("Original local coherence:", original_score)

    # Optional transformation function for readability
    def transform_score(raw):
        return 1.0 - math.exp(raw)

    print("Transformed original:", transform_score(original_score))

    # 2) Scramble the sentences multiple times and print scores
    num_trials = 20
    for i in range(num_trials):
        scrambled_text = scramble_sentences(original_text)
        scrambled_score = compute_coherence_probability(scrambled_text)
        print(f"\nTrial {i+1}:")
        #print("Scrambled text:", scrambled_text)
        print("Scrambled local coherence:", scrambled_score)
        print("Transformed scrambled:", transform_score(scrambled_score))


